from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import check_for_schedule, save_schedule, get_messages_as_openai_format, save_to_quiz_database
from langchain_core.chat_history import InMemoryChatMessageHistory, HumanMessage, AIMessage
import openai
from django.conf import settings

# OpenAI 클라이언트 초기화
openai.api_key = settings.OPENAI_API_KEY
chat_history = InMemoryChatMessageHistory()

class ChatAPIView(APIView):
    def post(self, request):
        user_input = request.data.get('user_input')
        if not user_input:
            return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        # 대화 종료 플래그 확인
        if request.data.get("end_conversation"):
            global chat_history
            chat_history = InMemoryChatMessageHistory()  # 대화 기록 초기화
            return Response({"message": "대화가 종료되었습니다."}, status=status.HTTP_200_OK)

        # 일정 확인 및 저장 여부 결정
        schedule_result = check_for_schedule(user_input)
        if schedule_result:
            ai_response = schedule_result["ai_response"]
            return Response({"ai_response": ai_response, "requires_confirmation": True})
        
        # 일정 저장 플래그 확인
        if request.data.get("save_schedule"):
            date_info = request.data.get("date_info")
            save_schedule(f"{date_info}: {user_input}")
            return Response({"ai_response": "일정이 저장되었어요!"})
        
        # 대화 히스토리에 사용자 메시지 추가
        chat_history.add_message(HumanMessage(content=user_input))

        # GPT 호출
        messages = get_messages_as_openai_format(chat_history)
        print(messages)  # 반환 값을 확인하기 위해 추가

        try:
            response = openai.Completion.create(
                model=settings.FINE_TUNED_MODEL_ID,
                prompt=messages,  # 구 버전에서는 prompt 파라미터를 사용
                temperature=0.4,
                max_tokens=150
            )
            ai_response = response.choices[0].text  # 구 버전에서는 text 속성을 사용

            # 대화 히스토리에 AI 메시지 추가
            chat_history.add_message(AIMessage(content=ai_response))

            # 대화 내용 저장
            save_to_quiz_database(user_input)

            return Response({"user_input": user_input, "ai_response": ai_response})
        except Exception as e:
            # 에러를 기록하고 클라이언트에게도 반환
            print(f"Error occurred: {e}")  # 콘솔에 에러 메시지 출력
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
