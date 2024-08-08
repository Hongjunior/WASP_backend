import re
import random
from datetime import datetime
from .models import Schedule, Conversation
from langchain_core.chat_history import InMemoryChatMessageHistory, HumanMessage, AIMessage

# 대화 메세지 형식 변환
def get_messages_as_openai_format(history):
    messages = []
    for message in history.messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        messages.append({"role": role, "content": message.content})
    return messages

# 일정 관련 키워드 탐지
def check_for_schedule(user_input):
    date_patterns = [
        r"(\d{1,2}월 \d{1,2}일)",      # "MM월 DD일"
        r"(\d{1,2}/\d{1,2})",          # "MM/DD"
        r"(\d{4}년 \d{1,2}월 \d{1,2}일)"  # "YYYY년 MM월 DD일"
    ]

    for pattern in date_patterns:
        match = re.search(pattern, user_input)
        if match:
            date_info = match.group()
            # 메시지 템플릿 리스트
            messages = [
                f"아 {date_info}에 일정이 있으시구나 잊지 않게 저장할까요?",
                f"{date_info}에 일정이 있으셔요? 잊지 않게 제가 저장해놓을까요?",
                f"{date_info}에 일정이 있어요? 제가 달력에 적어놓을까요?"
            ]
            # 랜덤으로 메시지 선택
            ai_response = random.choice(messages)

            # 응답 생성
            return {
                "ai_response": ai_response,
                "date_info": date_info
            }
    return None

# 일정 저장
def save_schedule(content):
    Schedule.objects.create(content=content)
    return "일정이 저장되었어요!"

# 사용자 발언 및 타임스탬프 저장
def save_to_quiz_database(user_input):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 현재 시간을 'YYYY-MM-DD HH:MM:SS' 형식으로 저장
    Conversation.objects.create(user_input=user_input, timestamp=timestamp)