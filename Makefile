PYTHON = python3
# Define the dataset name as a variable (defaults to 'test_run')
NAME ?= test_run

# Define paths based on the NAME
IMAGES = images/$(NAME)
PKL = data_pkl/$(NAME).pkl
DB = databases/$(NAME).db

# Parameters
RATIO = 0.75
MAX_FEATURES = 0
WINDOW_SIZE = 15
CLAHE = True
BLUR = False
# pass F for focal length required
POTRAIT ?= 0

FPS = 5

# Typing 'make' or 'make all' runs the whole pipeline in order
pre-all: preprocess features inject mapping clean_sparse

all: features inject mapping clean_sparse

preprocess:
	@echo "=== Preprocessing Images for $(NAME) ==="
	$(PYTHON) simple_preprocess.py --video $(NAME).mp4 --out_dir $(IMAGES) --fps $(FPS)

features:
	@echo "=== Processing Features for $(NAME) ==="
	$(PYTHON) clahe_window.py \
		--image_dir $(IMAGES) \
		--out_path $(PKL) \
		--ratio $(RATIO) \
		--max_features $(MAX_FEATURES) \
		--window_size $(WINDOW_SIZE) \
		--clahe $(CLAHE) \
		--blur $(BLUR)

inject:
	@echo "=== Injecting DB for $(NAME) ==="
	$(PYTHON) inject_data.py \
		--pkl $(PKL) \
		--db $(DB) \
		--f $(F) \
		--is_potrait $(POTRAIT)

mapping:
	@echo "=== Running Mapping for $(NAME) ==="
	$(PYTHON) run_mapping.py \
		--image_dir $(IMAGES) \
		--db $(DB) \
		--undistorted_dir undistorted/$(NAME) 


clean_sparse:
	@echo "=== Cleaning Sparse Models for $(NAME) ==="
	$(PYTHON) clean_sparse.py --input_dir undistorted/$(NAME)/0/sparse --output_dir cleaned_sparse/$(NAME)