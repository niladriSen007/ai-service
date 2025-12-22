package com.spring_ai.ai_service.service;

import lombok.RequiredArgsConstructor;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class AiService {

    private final ChatClient chatClient;
    private final EmbeddingModel embeddingModel;
    private final VectorStore vectorStore;

    public float[] getEmbeddings(String text) {
        return embeddingModel.embed(text);
    }

    public void inngestDataToVectorStore() {
        List<Document> documents = List.of(
               new Document(
                       "A thief who steals corporate secrets through the use of dream-sharing technology.",
                       Map.of("title", "Inception", "year", "2010","genre", "Sci-Fi")
               ),               new Document(
                       "A computer hacker learns from mysterious rebels about the true nature of his reality.",
                       Map.of("title", "The Matrix", "year", "1999","genre", "Sci-Fi")
               ),
                new Document(
                          "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
                          Map.of("title", "Interstellar", "year", "2014","genre", "Sci-Fi")
                )
        );
        vectorStore.add(documents);
    }

    public List<Document> getSimilaritySearch(String text){
            return vectorStore.similaritySearch(
                    SearchRequest.builder().query(text).topK(2).similarityThreshold(0.5).build()
            );
    }
}
