import argparse

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Display a white circle at a projector pixel on a black screen')
    parser.add_argument('x', type=int, help='projector pixel x coordinate')
    parser.add_argument('y', type=int, help='projector pixel y coordinate')
    parser.add_argument('--width', type=int, default=1920,
                        help='projector image width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                        help='projector image height (default: 1080)')
    parser.add_argument('--radius', type=int, default=10,
                        help='circle radius in pixels (default: 10)')
    parser.add_argument('--window-x', type=int, default=0,
                        help='desktop x position of the fullscreen window (default: 0)')
    parser.add_argument('--window-y', type=int, default=0,
                        help='desktop y position of the fullscreen window (default: 0)')
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        parser.error('width and height must be positive')
    if not 0 <= args.x < args.width or not 0 <= args.y < args.height:
        parser.error('circle coordinates must be inside the projector image')
    if args.radius <= 0:
        parser.error('radius must be positive')

    image = np.zeros((args.height, args.width), dtype=np.uint8)
    cv2.circle(image, (args.x, args.y), args.radius, 255, -1, cv2.LINE_8)

    window_name = 'Projector correspondence test'
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, args.window_x, args.window_y)
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, image)
        print(
            f'Displaying circle at projector pixel ({args.x}, {args.y}); '
            'press ESC or Q to close.')

        while True:
            key = cv2.waitKey(100) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
