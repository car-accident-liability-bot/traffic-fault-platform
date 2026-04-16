package com.trafficfault.backendapi.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.trafficfault.backendapi.api.dto.InferencePredictResponse;
import com.trafficfault.backendapi.config.TrafficInferenceProperties;
import com.trafficfault.backendapi.exception.InferenceServiceException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.InvalidMediaTypeException;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.util.StringUtils;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.multipart.MultipartFile;

@Component
public class TrafficInferenceClient {

    private final RestClient restClient;
    private final ObjectMapper objectMapper;
    private final TrafficInferenceProperties properties;

    public TrafficInferenceClient(
            RestClient trafficInferenceRestClient,
            ObjectMapper objectMapper,
            TrafficInferenceProperties properties
    ) {
        this.restClient = trafficInferenceRestClient;
        this.objectMapper = objectMapper;
        this.properties = properties;
    }

    public InferencePredictResponse predict(MultipartFile video, String questionType) {
        Path tempFile = null;

        try {
            tempFile = createTempVideo(video);

            MultiValueMap<String, Object> requestBody = new LinkedMultiValueMap<>();
            requestBody.add("video", buildVideoPart(video, tempFile));
            requestBody.add("question_type", questionType);

            InferencePredictResponse response = restClient.post()
                    .uri(properties.predictPath())
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(requestBody)
                    .retrieve()
                    .body(InferencePredictResponse.class);

            if (response == null || !StringUtils.hasText(response.answer())) {
                throw new InferenceServiceException("Python inference-service가 비어 있는 응답을 반환했습니다.");
            }

            return response;
        } catch (RestClientResponseException exception) {
            throw new InferenceServiceException(extractRemoteErrorMessage(exception.getResponseBodyAsString()), exception);
        } catch (ResourceAccessException exception) {
            throw new InferenceServiceException(
                    "Python inference-service에 연결할 수 없습니다. TRAFFIC_INFERENCE_BASE_URL 설정을 확인하세요.",
                    exception
            );
        } catch (IOException exception) {
            throw new InferenceServiceException("업로드 영상을 임시 파일로 처리하는 중 오류가 발생했습니다.", exception);
        } finally {
            deleteTempFileQuietly(tempFile);
        }
    }

    private Path createTempVideo(MultipartFile video) throws IOException {
        String originalFilename = StringUtils.hasText(video.getOriginalFilename())
                ? Objects.requireNonNull(video.getOriginalFilename())
                : "upload.mp4";

        String extension = extractExtension(originalFilename);
        Path tempFile = Files.createTempFile("traffic-backend-upload-", extension);

        // 수정 포인트:
        // getBytes()를 쓰지 않고 transferTo()로 디스크 임시 파일에 내려서
        // 큰 영상 업로드 시 힙 메모리 부담을 줄인다.
        video.transferTo(tempFile);
        return tempFile;
    }

    private HttpEntity<FileSystemResource> buildVideoPart(MultipartFile video, Path tempFile) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(resolveContentType(video.getContentType()));
        headers.setContentDispositionFormData("video", safeFilename(video.getOriginalFilename(), tempFile));

        return new HttpEntity<>(new FileSystemResource(tempFile), headers);
    }

    private MediaType resolveContentType(String contentType) {
        if (!StringUtils.hasText(contentType)) {
            return MediaType.APPLICATION_OCTET_STREAM;
        }

        try {
            return MediaType.parseMediaType(contentType);
        } catch (InvalidMediaTypeException exception) {
            return MediaType.APPLICATION_OCTET_STREAM;
        }
    }

    private String safeFilename(String originalFilename, Path tempFile) {
        if (!StringUtils.hasText(originalFilename)) {
            return tempFile.getFileName().toString();
        }

        return Path.of(originalFilename).getFileName().toString();
    }

    private String extractExtension(String filename) {
        int lastDotIndex = filename.lastIndexOf('.');
        if (lastDotIndex < 0) {
            return ".mp4";
        }
        return filename.substring(lastDotIndex);
    }

    private String extractRemoteErrorMessage(String responseBody) {
        if (!StringUtils.hasText(responseBody)) {
            return "Python inference-service 호출에 실패했습니다.";
        }

        try {
            JsonNode root = objectMapper.readTree(responseBody);
            if (root.hasNonNull("detail") && root.get("detail").asText().isBlank() == false) {
                return root.get("detail").asText();
            }
            if (root.hasNonNull("message") && root.get("message").asText().isBlank() == false) {
                return root.get("message").asText();
            }
            if (root.hasNonNull("error") && root.get("error").asText().isBlank() == false) {
                return root.get("error").asText();
            }
        } catch (Exception ignored) {
            // JSON 파싱 실패 시 원문 텍스트로 내려간다.
        }

        return responseBody;
    }

    private void deleteTempFileQuietly(Path tempFile) {
        if (tempFile == null) {
            return;
        }

        try {
            Files.deleteIfExists(tempFile);
        } catch (IOException ignored) {
            // 임시 파일 삭제 실패는 요청 전체를 실패로 보지 않는다.
        }
    }
}
