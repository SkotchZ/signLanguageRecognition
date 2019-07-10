import sys
import os
sys.path.append(os.path.join(__file__, "../../research"))
import cv2
import shutil


def main():
    cap = cv2.VideoCapture(0)
    remove_prev_data = False
    classes = os.listdir('../../prepared_data/test/A')
    classes.sort()
    saving_dir = '../../control_data'
    if os.path.exists(saving_dir) and remove_prev_data:
        shutil.rmtree(saving_dir)
    is_writing_mode = False
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    current_letter = "a"
    while(True):
        ret, frame = cap.read()
        key_value = cv2.waitKey(1)

        cv2.putText(frame, "Is writing mode: {}".format(str(is_writing_mode)),
                    (730, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Letter: {}".format(current_letter),
                    (730, 580), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (701, 701), (255, 0, 255), 1)
        cv2.imshow('frame', cv2.resize(frame, (640, 400)))

        if is_writing_mode and current_letter in classes:
            current_dir = os.path.join(saving_dir, current_letter)
            if not os.path.exists(current_dir):
                os.makedirs(current_dir)
            amount_of_files = str(len(os.listdir(current_dir)))
            croped = frame[1:700, 1:700]
            cv2.imwrite(os.path.join(current_dir, amount_of_files) + ".png",
                        croped)
            print(current_letter)

        if key_value == -1:
            continue

        if key_value == ord('-') | 128:
            is_writing_mode = False
            continue

        if key_value == ord('+') | 128:
            is_writing_mode = True
            continue

        current_letter = chr(key_value)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
