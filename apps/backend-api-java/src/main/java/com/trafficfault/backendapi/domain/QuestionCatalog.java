package com.trafficfault.backendapi.domain;

import com.trafficfault.backendapi.exception.BadRequestException;
import java.util.Arrays;
import java.util.List;
import org.springframework.util.StringUtils;

public enum QuestionCatalog {

    ROAD_ENV(
            "road-env",
            "해당 사고는 어떤 도로 환경에서 발생했는가?",
            "accident_place",
            "도로 환경"
    ),
    PLACE_FEATURE(
            "place-feature",
            "이 사고 장소의 특징은 무엇인가?",
            "accident_place_feature",
            "장소 특징"
    ),
    FAULT_RATIO(
            "fault-ratio",
            "이 사고의 과실비율은 어떻게 되는가?",
            "fault_ratio",
            "과실비율"
    ),
    MAJOR_FAULT(
            "major-fault",
            "과실비율 기준으로 더 큰 과실을 가진 차량은 누구인가?",
            "fault_compare",
            "책임 차량"
    );

    private final String frontendId;
    private final String questionTitle;
    private final String questionType;
    private final String shortLabel;

    QuestionCatalog(String frontendId, String questionTitle, String questionType, String shortLabel) {
        this.frontendId = frontendId;
        this.questionTitle = questionTitle;
        this.questionType = questionType;
        this.shortLabel = shortLabel;
    }

    public String frontendId() {
        return frontendId;
    }

    public String questionTitle() {
        return questionTitle;
    }

    public String questionType() {
        return questionType;
    }

    public String shortLabel() {
        return shortLabel;
    }

    public static QuestionCatalog fromInput(String rawValue) {
        if (!StringUtils.hasText(rawValue)) {
            throw new BadRequestException("question 값이 비어 있습니다.");
        }

        final String normalized = rawValue.trim();

        return Arrays.stream(values())
                .filter(item -> item.frontendId.equals(normalized)
                        || item.questionTitle.equals(normalized)
                        || item.questionType.equals(normalized))
                .findFirst()
                .orElseThrow(() -> new BadRequestException(
                        "지원하지 않는 question 값입니다. 지원 목록: "
                                + Arrays.stream(values())
                                .map(QuestionCatalog::questionTitle)
                                .toList()
                ));
    }

    public static List<QuestionCatalog> supportedQuestions() {
        return List.of(values());
    }
}
