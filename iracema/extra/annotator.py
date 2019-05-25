"""
GUI for audio annotation.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, MultiCursor # pylint: disable=import-error

import iracema
import iracema.segment as segment
import iracema.descriptors


class NoteAnnotator:
    """
    GUI for annotating note onsets and offsets for audio files.
    """

    def __init__(self, audio_file, output_csv, input_csv=None,
                 start_only=False):
        # loading audio file
        self.audio_file = audio_file
        self.audio = iracema.Audio(audio_file).normalize()

        # initializing segment list
        if input_csv:
            self.notes = segment.SegmentList.load_from_csv_file(
                self.audio, input_csv, limits='seconds', start_only=start_only)
        else:
            self.notes = segment.SegmentList([])

        self.output_csv = output_csv

        # playhead, onset_cursor and segment_spans
        self.fps = 10.
        self.i_playhead = 0
        self.playhead_positions = np.linspace(self.audio.start_time,
                                              self.audio.end_time,
                                              self.audio.duration * self.fps)
        self.onset_cursor = None
        self.segment_spans = []

        # state (boolean)
        self.state = 'waiting'  # {'waiting', 'playing', 'editing', 'editing-playing'}

        # plot waveform axes
        self.f = plt.figure(figsize=(16, 9), dpi=100)
        self.ax_waveform = self.f.add_subplot(211, picker=1)
        self.f.subplots_adjust(left=0.05, right=1 - 0.05)

        self.plot_waveform(self.audio)
        self.ax_waveform.set_xlim(self.audio.start_time, self.audio.end_time)
        self.ax_waveform.set_gid('ax1')

        # plot playhead and segment spans
        self.playhead = self.ax_waveform.axvline(
            x=self.playhead_positions[self.i_playhead], color='black')

        self.update_segment_spans()

        # create buttons
        ax_but_add_onset = self.f.add_subplot(425)
        self.but_add_onset = Button(ax_but_add_onset, 'add note onset')

        ax_but_add_offset = self.f.add_subplot(426)
        self.but_add_offset = Button(ax_but_add_offset, 'add note offset')

        ax_but_save = self.f.add_subplot(817)
        self.but_save = Button(ax_but_save, 'save to csv')

        ax_but_play = self.f.add_subplot(818)
        self.but_play = Button(ax_but_play, 'play / stop\n[space]')

        # connecting events
        self.but_play.on_clicked(self.play_or_stop)
        self.but_save.on_clicked(self.save_csv)
        self.but_add_onset.on_clicked(self.add_onset)
        self.but_add_offset.on_clicked(self.add_offset)
        self.f.canvas.mpl_connect('key_press_event',
                                  self.on_keypress)  # key presses
        self.f.canvas.mpl_connect('pick_event', self.on_pick)  # pick events

        # animation
        self.anim = animation.FuncAnimation(
            self.f, self.forward_playhead, interval=1000 / self.fps)

        # display figure
        self.f.canvas.manager.full_screen_toggle()

        plt.show()

    # axes plots
    def plot_waveform(self, audio):
        "Plot the waveform."
        rms = iracema.descriptors.rms(audio, 2048, 1024)  # TODO: fix hard code
        self.ax_waveform.plot(
            rms.time, rms.data, linewidth=1.2, alpha=0.9, color='blue')

    def plot_spectrogram(self, ax, spectrogram):
        "Plot the spectrogram."
        pass

    # callbacks
    def on_pick(self, event):
        print("artist: {} - {}, coords: {},{}".format(event.artist,
                                                      event.artist.get_gid(),
                                                      event.mouseevent.xdata,
                                                      event.mouseevent.ydata))

        if event.artist.get_gid() == 'ax1':
            if self.state == 'playing':
                self.audio.stop()
                self.state = 'waiting'
            self.update_playhead(event.mouseevent.xdata)

    def on_keypress(self, event):
        if event.key == ' ':
            self.play_or_stop(event)
        elif event.key == 'enter':
            self.stop_playing()
            self.reset_playhead()

    def add_onset(self, event):
        if self.state == 'waiting':
            self.state = 'editing'
            print('add onset on: ', self.playhead.get_xdata())
            self.onset_cursor = self.ax_waveform.axvline(
                x=self.playhead.get_xdata(), color='red', linewidth=2)

    def add_offset(self, event):
        if self.state == 'editing':
            self.state = 'waiting'
            onset = self.onset_cursor.get_xdata()[0]
            offset = self.playhead.get_xdata()
            self.onset_cursor.remove()

            new_note = segment.Segment(
                self.audio, onset, offset, limits='seconds')
            self.notes.append(new_note)
            self.update_segment_spans()

    def update_segment_spans(self):
        for span in self.segment_spans:
            span.remove()

        self.segment_spans = []

        for note in self.notes:
            note_span = self.ax_waveform.axvspan(
                xmin=note.start_time,
                xmax=note.end_time,
                edgecolor='black',
                facecolor='green',
                alpha=0.25,
                linewidth=1)
            self.segment_spans.append(note_span)

    def play_or_stop(self, event):
        if self.state == 'waiting':
            self.state = 'playing'
            self.audio.play_from_time(self.playhead_positions[self.i_playhead])
            plt.show()
        elif self.state == 'playing':
            self.state = 'waiting'
            self.audio.stop()
        elif self.state == 'editing':
            self.state = 'editing-playing'
            self.audio.play_from_time(self.playhead_positions[self.i_playhead])
        elif self.state == 'editing-playing':
            self.state = 'editing'
            self.audio.stop()

    def stop_playing(self):
        if self.state == 'playing':
            self.state = 'waiting'
            self.audio.stop()

    def reset_playhead(self):
        self.update_playhead(0)

    def update_playhead(self, x):
        self.playhead.set_xdata(x)
        self.i_playhead = int(x * self.fps)

    def forward_playhead(self, _):
        if self.state == 'playing' or self.state == 'editing-playing':
            self.i_playhead += 1
            # check if the playhead has reached the end of the audio
            if self.i_playhead == self.playhead_positions.shape[-1]: # pylint: disable=no-member
                self.i_playhead = 0
                self.state = 'waiting'
            self.playhead.set_xdata(self.playhead_positions[self.i_playhead])

    def save_csv(self, _):
        self.notes.save_to_csv_file(self.output_csv)
