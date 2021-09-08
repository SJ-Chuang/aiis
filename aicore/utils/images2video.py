import os, cv2, argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_dir', type=str, help='path to a directory of images.')
parser.add_argument('--fps', type=float, default=15.0)
parser.add_argument('-o', '--savename', help='path to save .mp4 file.', default=None)
args = parser.parse_args()

fourcc = cv2.VideoWriter_fourcc(*'MP4V')

savename = args.input_dir+'.mp4' if args.savename is None else args.savename
out = cv2.VideoWriter(savename, fourcc, 15.0, (1280, 720))

for i, file in enumerate(sorted(os.listdir(args.input_dir))):
    frame = cv2.imread(os.path.join(args.input_dir, file))
    out.write(frame)

out.release()