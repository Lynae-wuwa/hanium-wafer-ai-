import time

GPIO_AVAILABLE = False
GPIO = None

LED_PIN = 11       # Jetson 물리핀 11번
BUZZER_PIN = 13    # Jetson 물리핀 13번

_initialized = False


try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False
    GPIO = None


def init_gpio():
    global _initialized

    if not GPIO_AVAILABLE:
        return False, "Jetson.GPIO를 사용할 수 없어 소프트웨어 알람 상태만 기록합니다."

    if not _initialized:
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
        _initialized = True

    return True, "GPIO 초기화 완료"


def alarm_on():
    ok, msg = init_gpio()

    if not ok:
        return False, msg

    GPIO.output(LED_PIN, GPIO.HIGH)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)

    return True, "LED/부저 알람 ON"


def alarm_off():
    ok, msg = init_gpio()

    if not ok:
        return False, msg

    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

    return True, "LED/부저 알람 OFF"


def alarm_beep(seconds=1.0):
    ok, msg = init_gpio()

    if not ok:
        return False, msg

    GPIO.output(LED_PIN, GPIO.HIGH)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(seconds)
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

    return True, f"LED/부저 알람 {seconds}초 출력 완료"


def cleanup_gpio():
    global _initialized

    if GPIO_AVAILABLE and _initialized:
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        GPIO.cleanup()
        _initialized = False
