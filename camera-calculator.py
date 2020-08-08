"""Command line executable for the IPCV-Camera-Calculator software"""

import sys
import argparse
import cameraprocessor


def main():
    """
    This function is responsible for parsing the command line arguments
    provided by the user and initialize the camera-calculator accordingly
    """
    parser = argparse.ArgumentParser(
        description='IPCV-Camera-Calculator\n\n\
            A Computer Vision-based system for visually solving arithmetical expressions. It\
            works with pictures, pre-recorded videos or even live video streams (e.g. webcam).',
        epilog='The processing of a picture will happen istantaneously, while for either live\
            or pre-recorded videos the software will work in real-time, waiting until the user\
            has finished writing the expression before processing the frame.')

    parser.add_argument('-t', '--type', required=True,
                        choices=['image', 'video', 'webcam'], default='image',
                        help='The type of input media containing the arithmetical expression\
                            (image, video or webcam)')
    parser.add_argument('-p', '--path', required=True,
                        help='The path of the input media that has to be processed\
                            NOTE: for webcams, the path has to be an integer index (e.g. 1)')

    args = parser.parse_args(['--type', 'image', '--path', './foto/sottr1.jpg'])

    # Additional checks on argument correctness
    if args.type == 'webcam':
        try:
            args.path = int(args.path)
        except ValueError:
            print("for webcams, the path has to be an integer index (e.g. 1)")

    # Run the program with the parsed arguments
    cameraprocessor.run(args.type, args.path)


# Bootstrapper
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
