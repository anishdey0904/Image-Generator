
            const labels = document.querySelectorAll('.radio');

            labels.forEach(label => {
                label.addEventListener('click', function() {
                    labels.forEach(lbl => {
                        lbl.classList.remove('active'); // Remove active class
                    }); 
                    this.classList.add('active'); // Add active class to the clicked label
                });
            });

            const wave = document.querySelector(".wave");
            let clipPath = "";
      
            const N = 100;
            for (let i = 0; i < N + 1; i++) {
              clipPath =
                clipPath +
                `${(100 / N) * i}% ${
                  100 * (0.5 + 0.35 * Math.sin((2 * Math.PI * i) / N))
                }%,`;
            }
      
            clipPath = clipPath + "100% 100%,0 100%";
      
            clipPath = `polygon(${clipPath})`;
      
            wave.style["clip-path"] = clipPath;

            const current_tool = document.getElementById("current_tool"); // brush or eraser
            const brush_color = document.getElementById("brush_color");
            const brushSizeInput = document.getElementById("brush_size");

            const canvas = document.getElementById("drawing_area");
            const context = canvas.getContext("2d");

            let brush_size = 20; // default brush size
            let drawing = false;  // check whether the user is drawing inside canvas

            canvas.style.cursor = "url('static/images/brush.png') 9 55, crosshair"; // default brush cursor
            
            let curr_brush_color = "";  // default brush color is sky
            window.addEventListener("load", () => {
                curr_brush_color = "#87CEEB"; // default brush color is sky
                context.strokeStyle = curr_brush_color
            });

            let undoStack = [] // stack to store canvas states
            let redoStack = [] // stack to store undone canvas states
                       
            // GET CANVAS HEIGHT AND WIDTH BASED ON WINDOW HEIGHT AND WIDTH AND SIZE OF PARENT DIV ---->
            function resizeCanvas() {
                canvas.width = canvas.parentElement.clientWidth;
                canvas.height = canvas.parentElement.clientHeight;
            }
            window.addEventListener("resize", resizeCanvas);
            resizeCanvas();

            // GET BRUSH SIZE ---->
            brushSizeInput.addEventListener("input", (e) => {
                brush_size = e.target.value;
            });

            // GET BRUSH COLOR ---->
            brush_color.addEventListener("change", () => {
                const selectedColor = document.querySelector('input[name="color"]:checked').value;

                switch (selectedColor) {
                    case "sky":
                       curr_brush_color = "#87CEEB"; 
                        break;
                    case "clouds":
                        curr_brush_color = "#D3D3D3";
                        break;
                    case "grass":
                        curr_brush_color = "#32CD32";
                        break;
                    case "trees":
                        curr_brush_color = "#097017";
                        break;
                    case "river":
                        curr_brush_color = "#1C95FF";
                        break;
                    case "ocean":
                        curr_brush_color = "#0066BF";
                        break;
                    case "mountain":
                        curr_brush_color = "#944500";
                        break;
                    case "rocks":
                        curr_brush_color = "#6B4C32";
                        break;
                    case "sand":
                        curr_brush_color = "#FD7804";
                        break;
                    default:
                        curr_brush_color = "#000000";
                        break;
                }
                context.strokeStyle = curr_brush_color; // change the brush color
            });
            
            // UNDO - REDO FUNCTIONALITY  ---->
            function saveState(currStack) {
                currStack.push(canvas.toDataURL());
            }

            function redo() {
                if (redoStack.length > 0) {
                    const undoneState = redoStack.pop(); // get the previous undone state

                    const img = new Image();

                    img.src = undoneState; // Set the source of the image to the popped state
                    img.onload = function () {
                        context.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
                        context.drawImage(img, 0, 0, canvas.width, canvas.height); // Draw the state on canvas
                    };

                    saveState(undoStack); // save the current stae in undostack
                }
            }

            redoBtn.addEventListener("click", redo);           

            // Undo the last action
            function undo() {
                if (undoStack.length > 0) {
                    saveState(redoStack) // before undo put the current stack if undone state need to be redone
                    
                    const previousState = undoStack.pop(); // pop the last sate fro, stack
                    
                    const img = new Image(); // create an image object

                    img.src = previousState; // get the image from previous state
                    img.onload = function () {
                        context.clearRect(0, 0, canvas.width, canvas.height); // clear the canvas
                        context.drawImage(img, 0, 0); // put the prev image on the canvas
                    };
                }
            }

            undoBtn.addEventListener("click", undo);

            // DRAWING TOOL SELECTION FUNCTIONALITY ---->
            current_tool.addEventListener("change", function () {
                // Update the current tool based on the selection
                if (current_tool.value === "eraser") enableEraser();
                else disableEraser();
            });

            // Function to enable the eraser
            function enableEraser() {
                canvas.style.cursor = "url('static/images/eraser.png') 6 34, crosshair";
    
                // context.globalCompositeOperation = "destination-out"; // ***not wokring for undo erasing mode
                context.globalCompositeOperation = "source-over"; // normal drawing mode
                context.strokeStyle = "white"; 
                context.lineWidth = brush_size; // eraser size set to brush-size
            }

            // Function to enable the brush
            function disableEraser() {
                canvas.style.cursor = "url('static/images/brush.png') 9 55, crosshair";

                context.globalCompositeOperation = "source-over"; // normal drawing mode
                context.strokeStyle = curr_brush_color; // Change back to drawing color
                context.lineWidth = brush_size; // Reset to brush size
            }

            let notDrawing = 0;
            // FUNCTIONING THE CANVAS FOR DRAWING ---->
            // Start drawing when mouse is pressed down
            canvas.addEventListener("mousedown", (e) => {
                // before drawing or erasing put the current state in undostack
                saveState(undoStack);
                redoStack = []

                drawing = true;
                notDrawing = 0;
                context.lineWidth = brush_size;
            
                context.beginPath();  // Start a new path for drawing
                draw(e);  // Draw immediately when mouse is pressed
            });

            // Stop drawing when mouse is released
            canvas.addEventListener("mouseup", () => {
                
                drawing = false;
                notDrawing++;
                context.closePath();  // Close the current path
            });

            // Stop drawing when mouse-click is released outside the canvas
            window.addEventListener("mouseup", () => {
                drawing = false;
                context.closePath();  // Close the current path when mouse leaves
            });
            
            // If mouse re-enters the canvas and is currently drawing, start a new path from the entry point.
             canvas.addEventListener("mouseenter", (e) => {
                if (drawing) context.beginPath();
            });

            // Draw when mouse is moving over the canvas
            canvas.addEventListener("mousemove", (e) => {
                lastDrawTime = Date.now();
                draw(e);
            });

            function draw(e) {
                if(drawing == false) return;

                const canvasRect = canvas.getBoundingClientRect();
                const mouseX = e.clientX - canvasRect.left;
                const mouseY = e.clientY - canvasRect.top;

                // Drawing logic
                context.lineCap = "round";
                context.lineTo(mouseX, mouseY); // draw a line from prev coordinate to curr
                context.stroke(); // render the line on screen

                // start a new path from the current mouse position
                context.beginPath();
                context.moveTo(mouseX, mouseY); // move mouse to new coordinates
            }
            let isSending = false;
            let lastDrawTime = 0; // Track the last time the user drew on the canvas
            
            function sendCanvasInRealTime() {
                const currentTime = Date.now();
            
                // Check if the user is drawing or has not drawn in the last 1.5 seconds
                if (currentTime - lastDrawTime > 1500) return;
            
                if (isSending) return; // Prevent overlapping requests
                isSending = true;
            
                const image = canvas.toDataURL(); // Base64-encoded image
                fetch('http://127.0.0.1:5000/generate-image', {
                    method: 'POST',
                    body: JSON.stringify({ image: image }),
                    headers: {
                        'Content-Type': 'application/json',
                    },
                })
                    .then((response) => response.json())
                    .then((data) => {
                        if (data.image_url) {
                            const outputImg = document.getElementById('output_image');
                            // Cache-busting with a timestamp
                            outputImg.src = `http://127.0.0.1:5000${data.image_url}?t=${new Date().getTime()}`;
                        }
                    })
                    .catch((error) => console.error('Error:', error))
                    .finally(() => {
                        isSending = false;
                    });
            }
            
            // Trigger Backend Calls Periodically
            setInterval(sendCanvasInRealTime, 2500); // Check every 2.5 seconds
