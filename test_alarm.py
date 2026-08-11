import pygame
import time

pygame.mixer.init()

pygame.mixer.music.load("sounds/alarm.wav")

print("🔊 Alarm starting...")

pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    time.sleep(0.1)

pygame.mixer.quit()

print("Alarm finished.")
