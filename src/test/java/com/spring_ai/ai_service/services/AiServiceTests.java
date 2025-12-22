package com.spring_ai.ai_service.services;

import com.spring_ai.ai_service.service.AiService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class AiServiceTests {

    @Autowired
    private AiService aiService;

    @Test
    public void testStoreData() {
        aiService.inngestDataToVectorStore();
    }

    @Test
    public void testSimilaritySearch() {
        var res = aiService.getSimilaritySearch("movies of sci-fi genre");
        for(var doc: res) {
            System.out.println(doc);
        }

    }

}
