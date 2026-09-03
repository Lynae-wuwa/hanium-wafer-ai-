CLASS_NAMES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch"
]


PROCESS_GUIDE = {
    "Center": "증착 또는 식각 공정 중심부 균일도 이상 가능성 점검",
    "Donut": "포토 공정 또는 세정 공정의 원형 불균일 점검",
    "Edge-Loc": "웨이퍼 가장자리 로딩 또는 클램프 이상 점검",
    "Edge-Ring": "식각 공정 Edge ring 부품 또는 플라즈마 균일도 점검",
    "Loc": "국부 오염 또는 파티클 발생 장비 점검",
    "Near-full": "전체 공정 조건 이상 또는 장비 캘리브레이션 점검",
    "Random": "랜덤 파티클, 세정 공정, 이송 공정 점검",
    "Scratch": "웨이퍼 이송 장비 또는 핸들러 스크래치 점검"
}


def get_process_guide(label: str) -> str:
    return PROCESS_GUIDE.get(label, "등록되지 않은 불량 유형입니다. 추가 점검이 필요합니다.")
