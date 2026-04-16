package com.trafficfault.backendapi.api.dto;

import java.time.OffsetDateTime;

public record ErrorResponse(
        String message,
        OffsetDateTime timestamp
) {
    public static ErrorResponse of(String message) {
        return new ErrorResponse(message, OffsetDateTime.now());
    }
}
