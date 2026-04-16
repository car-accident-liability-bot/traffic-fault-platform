package com.trafficfault.backendapi.api.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record InferencePredictResponse(
        Boolean success,
        @JsonProperty("question_type") String questionType,
        String filename,
        String answer
) {
}
