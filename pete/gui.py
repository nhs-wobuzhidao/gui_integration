import threading
import time
import tkinter as tk
from CustomScrollBar import ScrollBar
from shared import *
from Step import Step
import cv2
from PIL import Image, ImageTk

current_step = 0
procedure = []
gui = None

def decision_logic():
    global procedure, current_step, gui
    while True:     # prevent calling before initialization
        if gui is not None: break

    while current_step < len(procedure):
        """
        Decision making frame goes here: we're simply calling validate() on the current step. Each step has its 
        own `validate()` method that is defined at initialization of the procedure. The crux of decision logic is 
        stored in each step, since each step has different criteria (unless there's some other way to implement it)
    
        (?) Methods that should be called here are 1) CV detect and 2) sensor detect. 
    
        :param data: a dict of CV data (bounding boxes) and sensor data. Or, evoke methods in main to get these data.
        """

        data = "ALPACAS"
        # data['CV'] = blah blah
        # data['sensor'] = blob blob
        # data['lala'] = wawa

        # TESTING ONLY: always validate to true after 10 seconds (lol)
        if procedure[current_step].validate(data):
            gui.mark_step_done(DONE)
        time.sleep(10)

class DisplayGUI:
    def __init__(self, app):
        """
        Initializations of the GUI components.
        :param app: a TK root object
        """
        global procedure, current_step

        self.app = app
        self.app.title("Project Pete")
        # "responsive" sizing
        min_width = int(self.app.winfo_screenwidth() * 0.85)
        min_height = int(self.app.winfo_screenheight() * 0.7)
        self.app.minsize(width=min_width, height=min_height)

        # Video Frame ========================================================
        # Create a frame for the live stream
        self.left_frame = tk.Frame(self.app, bg=dark_theme_background, width=0.7 * min_width)
        self.left_frame.pack(side="left", fill="both", expand=False)

        # Create a label to display stream
        self.livestream = tk.Label(self.left_frame, )
        self.livestream.pack(padx=(80, 50), pady=(40, 0))

        # Separate thread for live stream to not cause lagging
        update_thread = threading.Thread(target=self._update_frame)
        update_thread.daemon = True
        update_thread.start()

        # Logs ========================================================
        lw = int(self.left_frame.winfo_screenwidth() * 0.5)

        # Performance
        self.performance = tk.Frame(self.left_frame, bg=dark_theme_background, width=lw)
        self.performance.pack(side="left", fill="both", expand=False)
        self.performance_header = tk.Label(self.performance, text="Performance", bg=dark_theme_background,
                                           font=("Arial", 24, 'bold'))
        self.performance_header.pack(padx=(80, 0), pady=(10, 0))

        # Create a label to display the runtime
        self.runtime_label = tk.Label(self.performance, bg=dark_theme_background, text="Runtime: 0 seconds")
        self.runtime_label.pack(padx=(100, 0))

        # Create a thread for updating the runtime label
        runtime_thread = threading.Thread(target=self._update_runtime)
        runtime_thread.daemon = True
        runtime_thread.start()

        # Data
        self.data = tk.Frame(self.left_frame, width=lw, bg=dark_theme_background)
        self.data.pack(side="left", fill="both", expand=True)
        self.data_header = tk.Label(self.data, text="Data", bg=dark_theme_background, anchor='w',
                                    justify="left", font=("Arial", 24, 'bold'))
        self.data_header.pack(pady=(10, 0))

        # Procedure List ===================================================
        # Create a frame for the steps list (30% of total width)
        self.right_frame = tk.Frame(app, width=0.3 * min_width, bg=dark_theme_background)
        self.right_frame.pack(side="right", fill="both", expand=True)

        # Scrollable list view
        self.canvas = tk.Canvas(self.right_frame, bg=dark_theme_background, borderwidth=0, highlightthickness=0)
        self.procedure_list = tk.Frame(self.canvas, bg=dark_theme_background, borderwidth=0, highlightthickness=0)
        self.scrollbar = ScrollBar(self.procedure_list, orient="vertical", command=self.canvas.yview)
        self.procedure_list.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="top", fill="both", expand=True, pady=(40, 0))
        self.scrollbar.pack(side="left", fill="y", expand=False)
        self.procedure_list.pack(side="right", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.procedure_list, anchor="nw", tags="canvas_frame")

        procedure = self.get_procedure()
        # initialize lists in GUI
        for step in procedure:
            if step.status == IN_PROGRESS: current_step = step.index
            step.build(self.procedure_list)

        # Override button
        self.override = tk.Label(self.right_frame, fg='white', bg=override_button_color, text="Override - mark done",
                                 borderwidth=5)
        self.override.pack(fill="x", expand=False, padx=(10, 25), pady=(20, 10))
        self.override.bind("<ButtonRelease-1>", self.override_mark_done)

        # Decision logic ===================================
        # Separate thread for communicating with other functions to get decision result
        logic_thread = threading.Thread(target=decision_logic)
        logic_thread.daemon = True
        logic_thread.start()

    def get_procedure(self):
        """
        Gets steps in a procedure.
        Step is a defined class. More stuff can be added in there to help with validation.
        TODO: actual steps lol
        """
        global procedure, current_step

        # clear out and initialize procedure + step count
        procedure = []
        current_step = 0

        # dummy steps
        # TODO: define steps & their individual criteria
        for i in range(10):
            if i < 2:
                t = i
            elif i == 2:
                t = 2
            else:
                t = 3
            s = Step(index=i + 1, title=f"Step  {i + 1}", description="[wrench type], [other relevant info]",
                     status=list(colors.keys())[t])
            procedure.append(s)
        return procedure

    def mark_step_done(self, done_type):
        """
        Mark step as done and go to next step (if exists)
        :param done_type: DONE_OV (overriden), DONE (regular auto-approved)
        """
        global current_step, procedure

        isLastStep = current_step == len(procedure)
        procedure[current_step - 1].update_status(done_type, isFocus=isLastStep)

        self.canvas.yview_moveto(1.0)
        if isLastStep: return

        current_step += 1
        procedure[current_step - 1].update_status(IN_PROGRESS)

    def override_mark_done(self, e):
        """
        Overrides logic decision (mark as complete - OV)
        """
        self.mark_step_done(DONE_OV)

    def _update_frame(self):
        """
        Helper function to load a vid stream from camera
        """
        # cap = cv2.VideoCapture(live_stream_url)
        cap = cv2.VideoCapture(0,cv2.CAP_V4L2)  # camera
        if cap.isOpened():
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (750, 450))
                    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                    self.livestream.config(image=photo)
                    self.livestream.image = photo
        cap.release()

    def _update_runtime(self):
        """
        Helper function to update runtime clock
        """
        start_time = time.time()
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Format elapsed time into hh:mm:ss
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)

            runtime_str = f"Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}"
            self.runtime_label.config(text=runtime_str)

            time.sleep(1)  # Update the label every 1 second


if __name__ == '__main__':
    root = tk.Tk()
    gui = DisplayGUI(root)
    root.mainloop()

