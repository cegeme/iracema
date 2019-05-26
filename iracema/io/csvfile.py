"""
Methods for loading and writing data to CSV files.
"""


# TODO: this code is outdated, should be rewritten
def load_from_csv_file(time_series,
                       filename,
                       limits='seconds'):
    """
    Load info from a csv file into a list.
    """
    import csv
    start, end = [], []

    with open(filename, 'r', newline='') as ff:
        for row in csv.reader(ff, delimiter=','):
            start.append(np.float_(row[0]))
            if not start_only:
                end.append(np.float_(row[1]))

    segments = [
        cls(time_series, st, end, limits) for st, end in zip(start, end)
    ]
    return SegmentList(segments)


def save_to_csv_file(self, filename, limits='seconds'):
    """
    Save segment list to a csv file.
    """
    import csv

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for note in self:
            if limits == 'seconds':
                line = [note.start_time, note.end_time]
            else:
                line = [note.start, note.end]

            if start_only:
                writer.writerow([line[0]])
            else:
                writer.writerow(line)

