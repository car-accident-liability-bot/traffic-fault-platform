package com.trafficfault.backendapi.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "traffic.inference")
public record TrafficInferenceProperties(
        String baseUrl,
        String predictPath
) {
}
