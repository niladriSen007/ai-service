package com.spring_ai.ai_service.service;


import com.spring_ai.ai_service.advisor.TokenUsageAdvisor;
import lombok.RequiredArgsConstructor;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.SafeGuardAdvisor;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.client.advisor.vectorstore.VectorStoreChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class RAGService {

    private final VectorStore vectorStore;
    private final ChatClient chatClient;
    private final ChatMemory chatMemory;

    @Value("classpath:faq.pdf")
    Resource faqPdfResource;

    public String askAi(String question) {
        String template = """
                You are an AI assistant called Cody
                
                Rules:
                - Use ONLY the information provided in the context
                - You MAY rephrase, summarize, and explain in natural language
                - Do NOT introduce new concepts or facts
                - If multiple context sections are relevant, combine them into a single explanation.
                - If the answer is not present, say "I don't know"
                
                Context:
                {context}
                
                Answer in a friendly, conversational tone.
                """;
        List<Document> documents = vectorStore.similaritySearch(
                SearchRequest.builder().query(question).topK(2).similarityThreshold(0.3).build()
        );
        String context = documents.stream().map(Document::getText).collect(Collectors.joining("\n\n"));
        PromptTemplate promptTemplate = new PromptTemplate(template);
        String SYSTEM_PROMPT = promptTemplate.render(Map.of("context", context));
        return chatClient.prompt().system(SYSTEM_PROMPT).user(question).call().content();

    }

    public void inngestPdfToVectorStore() {
        PagePdfDocumentReader pdfReader = new PagePdfDocumentReader(faqPdfResource);
        List<Document> documents = pdfReader.read();

        TokenTextSplitter textSplitter = new TokenTextSplitter(200, 50, 5, 1000, true);
        List<Document> chunks = textSplitter.apply(documents);

        vectorStore.add(chunks);
    }

    public String askAiAboutPdf(String question) {
        String template = """
                You are an AI assistant called Pody
                
                Rules:
                - Use ONLY the information provided in the context
                - You MAY rephrase, summarize, and explain in natural language
                - Do NOT introduce new concepts or facts
                - If multiple context sections are relevant, combine them into a single explanation.
                - If the answer is not present, say "I don't know"
                
                Context:
                {context}
                
                Answer in a friendly, conversational tone and nice gesture.
                """;

        List<Document> documents = vectorStore.similaritySearch(
                SearchRequest.builder().query(question).topK(2).filterExpression("file_name == 'faq.pdf'").similarityThreshold(0.3).build()
        );
        String context = documents.stream().map(Document::getText).collect(Collectors.joining("\n\n"));
        PromptTemplate promptTemplate = new PromptTemplate(template);
        String SYSTEM_PROMPT = promptTemplate.render(Map.of("context", context));
        return chatClient.prompt().system(SYSTEM_PROMPT).user(question).call().content();
    }

    public String askAiByAdvisor(String question, String userId) {
        return chatClient.prompt()
                .system("""
                            You are a helpful assistant.
                            Answer the question based on the context provided.
                            If the answer is not present, say "I don't know".
                            Your name is Pody and if you know the user name from the previous context then greet the user with his name.
                        """)
                .user(question)
                .advisors(
                        SafeGuardAdvisor.builder().sensitiveWords(List.of(
                                // case sensitive
                                "password", "ssn", "credit card", "address", "politics", "religion", "hate speech"
                        )).build(),
                        TokenUsageAdvisor.builder().build(),
                        MessageChatMemoryAdvisor.builder(chatMemory)
                                .conversationId(userId)
                                .build(),
                        VectorStoreChatMemoryAdvisor.builder(vectorStore)
                                .conversationId(userId)
                                .build(),
                        QuestionAnswerAdvisor.builder(vectorStore)
                                .searchRequest(
                                        SearchRequest.builder()
                                                .filterExpression("file_name == 'faq.pdf'")
                                                .topK(3)
                                                .build()
                                )
                                .build()
                )
                .call()
                .content();
    }

}
