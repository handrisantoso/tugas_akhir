import cv2
import json
import os
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog

class Annotator:
    def __init__(self, video_path=None, output_file=None):
        # Allow selecting a video file if none provided
        if video_path is None:
            video_path = filedialog.askopenfilename(title="Select Video", 
                                                filetypes=[("Video files", "*.mp4 *.mkv *.webm")])
            if not video_path:
                print("No video selected. Exiting.")
                return
        
        # Allow selecting/creating an output file if none provided
        if output_file is None:
            output_file = filedialog.asksaveasfilename(title="Save Annotations As", 
                                                    defaultextension=".json",
                                                    filetypes=[("JSON files", "*.json")])
            if not output_file:
                output_file = os.path.splitext(video_path)[0] + "_annotations.json"
        
        self.video_path = video_path
        self.output_file = output_file
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
            
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.sequence_start = None
        self.annotations = {}

        # Load existing annotations if file exists
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                self.annotations = json.load(f)

        # Jump to the latest annotated frame if available
        if self.annotations:
            self.current_frame = max(seq["end_frame"] for seq in self.annotations.values())
        else:
            self.current_frame = 0

        # Set up GUI
        self.root = Tk()
        self.root.title("Tekken Move Annotator")
        
        # Display frame
        self.display_frame = Frame(self.root)
        self.display_frame.pack(side="top", fill="both", expand=True)
        
        self.canvas = Canvas(self.display_frame, width=480, height=270)
        self.canvas.pack()
        
        # Slider frame with better controls
        slider_frame = Frame(self.root)
        slider_frame.pack(fill="x", padx=10, pady=5)
        
        # Add time display and frame info
        time_frame = Frame(slider_frame)
        time_frame.pack(fill="x", pady=2)
        
        self.time_label = Label(time_frame, text="00:00 / 00:00")
        self.time_label.pack(side="left")
        
        # Frame counter and current sequence duration
        self.frame_info_label = Label(time_frame, text="Frame: 0/0")
        self.frame_info_label.pack(side="right")
        
        # Current sequence frame count (when recording)
        self.sequence_frames_label = Label(time_frame, text="", fg="red", font=("Arial", 8, "bold"))
        self.sequence_frames_label.pack(side="right", padx=10)

        SLIDER_WIDTH = 600
        self.mark_canvas = Canvas(slider_frame, width=SLIDER_WIDTH, height=10, bg="white", highlightthickness=0)
        self.mark_canvas.pack(fill="x", padx=0, pady=(0, 2))

        self.frame_slider = Scale(slider_frame, from_=0, to=self.total_frames-1,
                                orient="horizontal", length=SLIDER_WIDTH, 
                                command=self.update_frame,
                                resolution=1, tickinterval=self.total_frames//10)
        self.frame_slider.pack(fill="x", padx=0)
        
        # Jump controls with frame input
        jump_frame = Frame(slider_frame)
        jump_frame.pack(fill="x", pady=5)
        
        Button(jump_frame, text="<<< 1 sec", command=lambda: self.jump_frames(-int(self.fps))).pack(side="left")
        Button(jump_frame, text="<< 10 frames", command=lambda: self.jump_frames(-10)).pack(side="left")
        Button(jump_frame, text="< 1 frame", command=lambda: self.jump_frames(-1)).pack(side="left")
        Button(jump_frame, text="1 frame >", command=lambda: self.jump_frames(1)).pack(side="left")
        Button(jump_frame, text="10 frames >>", command=lambda: self.jump_frames(10)).pack(side="left")
        Button(jump_frame, text="1 sec >>>", command=lambda: self.jump_frames(int(self.fps))).pack(side="left")
        
        # Frame jump input
        jump_input_frame = Frame(jump_frame)
        jump_input_frame.pack(side="right", padx=10)
        Label(jump_input_frame, text="Jump to frame:").pack(side="left")
        self.jump_entry = Entry(jump_input_frame, width=8)
        self.jump_entry.pack(side="left", padx=2)
        self.jump_entry.bind('<Return>', self.jump_to_frame)
        Button(jump_input_frame, text="Go", command=self.jump_to_frame).pack(side="left", padx=2)

        # Status indicator
        self.status_frame = Frame(self.root, height=30)
        self.status_frame.pack(side="top", fill="x")
        self.status_label = Label(self.status_frame, text="Ready", fg="black", bg="lightgray")
        self.status_label.pack(fill="x")
        
        # ...existing code...
        
        # Dual-character input UI
        self.input_frame = Frame(self.root)
        self.input_frame.pack(side="top", fill="x", padx=10, pady=5)
        
        # Character setup frame
        char_setup_frame = Frame(self.input_frame)
        char_setup_frame.pack(fill="x", pady=5)
        
        Label(char_setup_frame, text="Player 1 (Left):").pack(side="left")
        self.p1_char_entry = Entry(char_setup_frame, width=10)
        self.p1_char_entry.pack(side="left", padx=5)
        
        Label(char_setup_frame, text="Player 2 (Right):").pack(side="left")
        self.p2_char_entry = Entry(char_setup_frame, width=10)
        self.p2_char_entry.pack(side="left", padx=5)
        
        # Annotation input frame
        annotation_frame = Frame(self.input_frame)
        annotation_frame.pack(fill="x", pady=5)
        
        # Player 1 annotation
        p1_frame = Frame(annotation_frame)
        p1_frame.pack(side="left", fill="x", expand=True, padx=5)
        
        Label(p1_frame, text="Player 1 Move:").pack()
        self.p1_move_entry = Entry(p1_frame, width=15)
        self.p1_move_entry.pack(fill="x")
        # Bind key release to check for previous annotations
        self.p1_move_entry.bind('<KeyRelease>', self.check_move_history)
        
        # VS separator
        vs_frame = Frame(annotation_frame)
        vs_frame.pack(side="left", padx=10)
        Label(vs_frame, text="VS", font=("Arial", 12, "bold")).pack(pady=20)
        
        # Player 2 annotation
        p2_frame = Frame(annotation_frame)
        p2_frame.pack(side="left", fill="x", expand=True, padx=5)
        
        Label(p2_frame, text="Player 2 Move:").pack()
        self.p2_move_entry = Entry(p2_frame, width=15)
        self.p2_move_entry.pack(fill="x")
        # Bind key release to check for previous annotations
        self.p2_move_entry.bind('<KeyRelease>', self.check_move_history)
        
        # Move history display frame
        history_frame = Frame(self.input_frame)
        history_frame.pack(fill="x", pady=5)
        
        self.history_label = Label(history_frame, text="", fg="blue", font=("Arial", 8), 
                                 wraplength=800, justify="left")
        self.history_label.pack(fill="x")
        
        # Quick buttons for common states
        quick_frame = Frame(self.input_frame)
        quick_frame.pack(fill="x", pady=5)
        
        Button(quick_frame, text="P1 Active / P2 Idle", 
               command=lambda: self.set_quick_state("active", "idle")).pack(side="left", padx=2)
        Button(quick_frame, text="P1 Idle / P2 Active", 
               command=lambda: self.set_quick_state("idle", "active")).pack(side="left", padx=2)
        Button(quick_frame, text="Both Idle", 
               command=lambda: self.set_quick_state("idle", "idle")).pack(side="left", padx=2)
        Button(quick_frame, text="Clear Both", 
               command=lambda: self.set_quick_state("", "")).pack(side="left", padx=2)
        
        # Controls
        self.btn_frame = Frame(self.root)
        self.btn_frame.pack(side="bottom", fill="x")
        
        Button(self.btn_frame, text="Prev Frame (←)", command=self.prev_frame).pack(side="left")
        Button(self.btn_frame, text="Next Frame (→)", command=self.next_frame).pack(side="left")
        self.start_btn = Button(self.btn_frame, text="Start Sequence (Control+s)", command=self.start_sequence)
        self.start_btn.pack(side="left")
        self.cancel_btn = Button(self.btn_frame, text="Cancel Sequence (Control+c)", command=self.cancel_sequence)
        self.cancel_btn.pack(side="left")
        Button(self.btn_frame, text="Save (Command+S)", command=self.save_annotations).pack(side="right")
        
        # Key bindings
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('<Shift-Left>', lambda e: self.jump_frames(-10))
        self.root.bind('<Shift-Right>', lambda e: self.jump_frames(10))
        self.root.bind('<Control-s>', lambda e: self.start_sequence())
        self.root.bind('<Control-c>', lambda e: self.cancel_sequence())
        self.root.bind('<Command-s>', lambda e: self.save_annotations())
        
        # Load the first frame
        self.draw_annotation_marks()
        self.update_frame(self.current_frame)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def jump_to_frame(self, event=None):
        """Jump to a specific frame number entered by user"""
        try:
            # Use 0-based indexing directly
            target_frame = int(self.jump_entry.get())
            target_frame = max(0, min(target_frame, self.total_frames - 1))
            self.update_frame(target_frame)
            self.jump_entry.delete(0, END)  # Clear the input
        except ValueError:
            self.status_label.config(text="Invalid frame number!", bg="orange", fg="black")

    def check_move_history(self, event=None):
        """Check if the current move combination has been annotated before"""
        p1_move = self.p1_move_entry.get().strip().lower()
        p2_move = self.p2_move_entry.get().strip().lower()
        
        if not p1_move and not p2_move:
            self.history_label.config(text="")
            return
        
        # Find matching annotations
        matches = []
        for seq_id, seq_data in self.annotations.items():
            p1_annotated = seq_data["move"].lower() if seq_data.get("player") == "player1" else ""
            p2_annotated = seq_data["move"].lower() if seq_data.get("player") == "player2" else ""
            
            p1_match = p1_move and (p1_move == p1_annotated)
            p2_match = p2_move and (p2_move == p2_annotated)
            
            if p1_match or p2_match:
                duration = seq_data["duration_frames"]
                # Use 0-based frame numbers
                start_frame = seq_data["start_frame"]
                end_frame = seq_data["end_frame"] + 1
                matches.append({
                    'id': seq_id,
                    'p1_move': seq_data["player1"]["move"],
                    'p2_move': seq_data["player2"]["move"],
                    'duration': duration,
                    'start': start_frame,
                    'end': end_frame
                })
        
        if matches:
            history_text = "Previous annotations: "
            history_items = []
            for match in matches[:3]:  # Show only first 3 matches to avoid clutter
                history_items.append(f"{match['p1_move']} vs {match['p2_move']} ({match['duration']} frames, {match['start']}-{match['end']})")
            
            history_text += " | ".join(history_items)
            if len(matches) > 3:
                history_text += f" ... and {len(matches) - 3} more"
            
            self.history_label.config(text=history_text)
        else:
            self.history_label.config(text="")

    def set_quick_state(self, p1_state, p2_state):
        """Set quick states for both players"""
        self.p1_move_entry.delete(0, END)
        self.p1_move_entry.insert(0, p1_state)
        self.p2_move_entry.delete(0, END)
        self.p2_move_entry.insert(0, p2_state)
        # Trigger history check after setting states
        self.check_move_history()

    def update_frame(self, frame_index=None):
        if not self.cap.isOpened():
            return
            
        if frame_index is not None:
            self.current_frame = int(float(frame_index))
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        
        if not ret:
            return
            
        # Convert frame to PIL Image and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((480, 270), Image.LANCZOS)
        
        # Convert to PhotoImage and display
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        
        # Update slider to match current position
        self.frame_slider.set(self.current_frame)
        
        # Update time display
        current_time = self.current_frame / self.fps
        total_time = self.total_frames / self.fps
        self.time_label.config(text=f"{self.format_time(current_time)} / {self.format_time(total_time)}")
        
        # Update frame info - use 0-based frame numbers
        self.frame_info_label.config(text=f"Frame: {self.current_frame}/{self.total_frames-1}")
        
        # Update sequence frame count if recording
        if self.sequence_start is not None:
            sequence_duration = self.current_frame - self.sequence_start + 1
            self.sequence_frames_label.config(text=f"Recording: {sequence_duration} frames")
        else:
            self.sequence_frames_label.config(text="")
        
        # Update title with recording status - use 0-based frame numbers
        title = f"Tekken Move Annotator - Frame {self.current_frame}/{self.total_frames-1}"
        if self.sequence_start is not None:
            title += f" - RECORDING (from frame {self.sequence_start})"
        self.root.title(title)
        
        # Update status based on recording state
        if self.sequence_start is not None:
            sequence_duration = self.current_frame - self.sequence_start + 1
            self.status_label.config(text=f"Recording from frame {self.sequence_start} | Current duration: {sequence_duration} frames", bg="red", fg="white")
        else:
            self.status_label.config(text="Ready", bg="lightgray", fg="black")
    
    def format_time(self, seconds):
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def jump_frames(self, offset):
        """Jump a specific number of frames forward or backward"""
        target_frame = self.current_frame + offset
        target_frame = max(0, min(target_frame, self.total_frames - 1))
        self.update_frame(target_frame)
    
    def prev_frame(self, event=None):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_frame()
    
    def next_frame(self, event=None):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.update_frame()
    
    def on_close(self):
        """Handle window close event"""
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def start_sequence(self, event=None):
        if self.sequence_start is None:
            self.sequence_start = self.current_frame
            self.start_btn.config(bg="red", fg="white", text="End Sequence (s)")
            self.update_frame()
        else:
            self.end_sequence()
    
    def cancel_sequence(self, event=None):
        self.sequence_start = None
        self.start_btn.config(bg="SystemButtonFace", fg="black", text="Start Sequence (s)")
        self.update_frame()
    
    def draw_annotation_marks(self):
        """Draw marks on the canvas for annotated sequences."""
        self.mark_canvas.delete("all")
        width = int(self.mark_canvas['width'])
        for seq in self.annotations.values():
            start = seq["start_frame"]
            end = seq["end_frame"]
            # Map frame to canvas position
            x1 = int(start / (self.total_frames - 1) * width)
            x2 = int(end / (self.total_frames - 1) * width)
            # Draw a small rectangle or line for the sequence
            self.mark_canvas.create_rectangle(x1, 0, x2, 10, fill="#4caf50", outline="")

    def end_sequence(self):
        p1_character = self.p1_char_entry.get().strip()
        p2_character = self.p2_char_entry.get().strip()
        p1_move = self.p1_move_entry.get().strip()
        p2_move = self.p2_move_entry.get().strip()
    
        saved_any = False
        status_msgs = []
    
        if p1_character and p1_move:
            sequence_id = f"p1_sequence_{len(self.annotations) + 1}"
            self.annotations[sequence_id] = {
                "player": "player1",
                "character": p1_character,
                "move": p1_move,
                "start_frame": self.sequence_start,
                "end_frame": self.current_frame,
                "start_time": self.sequence_start / self.fps,
                "end_time": self.current_frame / self.fps,
                "duration_frames": self.current_frame - self.sequence_start + 1,
                "annotation_type": "single_character"
            }
            status_msgs.append(f"{p1_character}: {p1_move} (frames {self.sequence_start}-{self.current_frame})")
            saved_any = True
    
        if p2_character and p2_move:
            sequence_id = f"p2_sequence_{len(self.annotations) + 1}"
            self.annotations[sequence_id] = {
                "player": "player2",
                "character": p2_character,
                "move": p2_move,
                "start_frame": self.sequence_start,
                "end_frame": self.current_frame,
                "start_time": self.sequence_start / self.fps,
                "end_time": self.current_frame / self.fps,
                "duration_frames": self.current_frame - self.sequence_start + 1,
                "annotation_type": "single_character"
            }
            status_msgs.append(f"{p2_character}: {p2_move} (frames {self.sequence_start}-{self.current_frame})")
            saved_any = True
    
        if saved_any:
            self.status_label.config(text=" | ".join(status_msgs), bg="green", fg="white")
            self.p1_move_entry.delete(0, END)
            self.p2_move_entry.delete(0, END)
        else:
            self.status_label.config(text="Error: Enter character and move for at least one player!", bg="orange", fg="black")
    
        self.sequence_start = None
        self.start_btn.config(bg="SystemButtonFace", fg="black", text="Start Sequence (s)")
        self.draw_annotation_marks()
        self.update_frame()
    
    def save_annotations(self, event=None):
        with open(self.output_file, "w") as f:
            json.dump(self.annotations, f, indent=2)
        self.status_label.config(text=f"Annotations saved to {self.output_file}", bg="blue", fg="white")
        self.draw_annotation_marks()
        print(f"Annotations savesd to {self.output_file}")

if __name__ == "__main__":
    Annotator()
    #Annotator("move_list/bryan/frames", "move_list/bryan/bryan_movelist.json")