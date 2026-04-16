package com.trafficfault.backendapi.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;

@Configuration
public class RestClientConfig {

    @Bean
    public RestClient trafficInferenceRestClient(
            RestClient.Builder builder,
            TrafficInferenceProperties properties
    ) {
        return builder
                .baseUrl(properties.baseUrl())
                .build();
    }
}
