from django.test import TestCase
from django.urls import reverse

class ChatAPITest(TestCase):
    def test_schedule_detection(self):
        response = self.client.post(reverse('chat'), {'user_input': '12월 25일에 약속이 있어요.'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('일정이 있으시구나', response.json()['ai_response'])

    def test_save_schedule(self):
        self.client.post(reverse('chat'), {'user_input': '12월 25일에 약속이 있어요.'})
        response = self.client.post(reverse('chat'), {
            'user_input': '응',
            'save_schedule': True,
            'date_info': '12월 25일'
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['ai_response'], '일정이 저장되었어요!')