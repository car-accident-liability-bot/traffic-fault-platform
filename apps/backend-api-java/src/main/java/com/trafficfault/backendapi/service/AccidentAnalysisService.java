package com.trafficfault.backendapi.service;

import com.trafficfault.backendapi.api.dto.AnalyzeResponse;
import com.trafficfault.backendapi.api.dto.InferencePredictResponse;
import com.trafficfault.backendapi.domain.QuestionCatalog;
import com.trafficfault.backendapi.exception.BadRequestException;
import java.util.Set;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

@Service
public class AccidentAnalysisService {

    private static final Set<String> ALLOWED_VIDEO_EXTENSIONS = Set.of(
            ".mp4", ".avi", ".mov", ".mkv", ".webm"
    );

    private final TrafficInferenceClient trafficInferenceClient;

    public AccidentAnalysisService(TrafficInferenceClient trafficInferenceClient) {
        this.trafficInferenceClient = trafficInferenceClient;
    }

    public AnalyzeResponse analyze(MultipartFile video, String question) {
        validateVideo(video);

        QuestionCatalog selectedQuestion = QuestionCatalog.fromInput(question);
        InferencePredictResponse inferenceResponse = trafficInferenceClient.predict(
                video,
                selectedQuestion.questionType()
        );

        return new AnalyzeResponse(
                true,
                selectedQuestion.questionTitle(),
                selectedQuestion.questionType(),
                StringUtils.hasText(inferenceResponse.filename()) ? inferenceResponse.filename() : video.getOriginalFilename(),
                inferenceResponse.answer()
        );
    }

    private void validateVideo(MultipartFile video) {
        if (video == null || video.isEmpty()) {
            throw new BadRequestException("업로드된 영상 파일이 없습니다.");
        }

        String originalFilename = video.getOriginalFilename();
        if (!StringUtils.hasText(originalFilename)) {
            throw new BadRequestException("영상 파일명이 비어 있습니다.");
        }

        int lastDotIndex = originalFilename.lastIndexOf('.');
        if (lastDotIndex < 0) {
            throw new BadRequestException("지원하지 않는 비디오 형식입니다.");
        }

        String extension = originalFilename.substring(lastDotIndex).toLowerCase();
        if (!ALLOWED_VIDEO_EXTENSIONS.contains(extension)) {
            throw new BadRequestException("지원하지 않는 비디오 형식입니다.");
        }
    }
}
