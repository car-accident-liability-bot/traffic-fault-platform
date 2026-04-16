package com.trafficfault.backendapi.api.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record QuestionOptionResponse(
        String id,
        String title,
        @JsonProperty("question_type") String questionType,
        String shortLabel
) {
}
