package com.trafficfault.backendapi;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class BackendApiJavaApplication {

    public static void main(String[] args) {
        SpringApplication.run(BackendApiJavaApplication.class, args);
    }
}
