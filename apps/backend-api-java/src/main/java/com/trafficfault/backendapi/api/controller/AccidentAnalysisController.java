package com.trafficfault.backendapi.api.controller;

import com.trafficfault.backendapi.api.dto.AnalyzeResponse;
import com.trafficfault.backendapi.api.dto.QuestionOptionResponse;
import com.trafficfault.backendapi.domain.QuestionCatalog;
import com.trafficfault.backendapi.service.AccidentAnalysisService;
import java.util.List;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api")
public class AccidentAnalysisController {

    private final AccidentAnalysisService accidentAnalysisService;

    public AccidentAnalysisController(AccidentAnalysisService accidentAnalysisService) {
        this.accidentAnalysisService = accidentAnalysisService;
    }

    @GetMapping("/health")
    public Object health() {
        return java.util.Map.of(
                "status", "ok",
                "service", "backend-api-java"
        );
    }

    @GetMapping("/questions")
    public List<QuestionOptionResponse> questions() {
        return QuestionCatalog.supportedQuestions().stream()
                .map(item -> new QuestionOptionResponse(
                        item.frontendId(),
                        item.questionTitle(),
                        item.questionType(),
                        item.shortLabel()
                ))
                .toList();
    }

    @PostMapping(
            path = "/analyze",
            consumes = MediaType.MULTIPART_FORM_DATA_VALUE,
            produces = MediaType.APPLICATION_JSON_VALUE
    )
    public AnalyzeResponse analyze(
            @RequestParam("video") MultipartFile video,
            @RequestParam("question") String question
    ) {
        return accidentAnalysisService.analyze(video, question);
    }
}
