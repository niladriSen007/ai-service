package com.spring_ai.ai_service.services;

import com.spring_ai.ai_service.service.AiService;
import com.spring_ai.ai_service.service.RAGService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class RagServiceTests {
    @Autowired
    private AiService aiService;
    @Autowired
    private RAGService ragService;

    @Test
    public void testStoreData() {
        aiService.inngestDataToVectorStore();
    }

    @Test
    public void testAskAI() {
        String res = ragService.askAi("What is Spring ai service?");
        System.out.println(res);
    }

    @Test
    public void inngestPdfDataToVectorStore() {
        ragService.inngestPdfToVectorStore();
    }

    @Test
    public void askAiAboutPdf() {
        String res = ragService.askAiAboutPdf("I am not able to login so what to do now");
        System.out.println(res);
    }
}
