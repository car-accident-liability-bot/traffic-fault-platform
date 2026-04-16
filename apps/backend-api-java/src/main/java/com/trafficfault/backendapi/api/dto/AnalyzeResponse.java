package com.trafficfault.backendapi.api.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record AnalyzeResponse(
        boolean success,
        String question,
        @JsonProperty("question_type") String questionType,
        String filename,
        String answer
) {
}
