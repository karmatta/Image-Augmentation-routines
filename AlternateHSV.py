hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
shift_h = (h + 90) % 180
shift_s = (s + 90) % 180
shift_v = (s + 90) % 180
shift_hsv = cv2.merge([shift_h, shift_s, shift_v])
shift_img = cv2.cvtColor(shift_hsv, cv2.COLOR_HSV2BGR)