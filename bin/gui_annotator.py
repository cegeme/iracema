#!/usr/bin/env python3

from iracema.extra.annotator import NoteAnnotator

import argparse
import warnings
warnings.filterwarnings('ignore')

p = argparse.ArgumentParser(description="""
Graphical utility for manual note onset/offset annotation.
""")

p.add_argument('audio_file', help='audio file to be loaded')
p.add_argument(
    '-o',
    dest='output_csv',
    action='store',
    default=None,
    help="""name of the output csv file where the segment data will be saved
    (if this parameter is not specified, the name of the output csv will be the
    name of the audio file appended with the suffix \'.notes.csv\')""")
p.add_argument(
    '-i',
    dest='input_csv',
    action='store',
    default=None,
    help='csv file with the segmentation data to be loaded')
p.add_argument(
    '--start-only',
    action='store_true',
    default=False,
    help='load the start (onset) only')

args = p.parse_args()

print(args)

if not args.output_csv:
    args.output_csv = args.audio_file + '.notes.csv'

NoteAnnotator(
    args.audio_file,
    output_csv=args.output_csv,
    input_csv=args.input_csv,
    start_only=args.start_only)
