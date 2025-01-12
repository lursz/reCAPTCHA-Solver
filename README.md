# reCAPTCHA Solver
![captch_checkbox](https://i.gifer.com/ARda.gif)
# About
CAPTCHA is an acronym for Completely Automated Turing Test to Tell Computers and Humans Apart. As the name suggests, it is a system designed solely to distinguish between humans and machines. This means that CAPTCHA challenges ought to be easily solvable for humans, but impossible to solve for any automated systems.

This repository explores the feasibility of breaking image-based reCAPTCHA challenges using machine learning techniques. The aim is to proove that reCAPTCHA is no longer safe, and can be broken with relatively mild effort, even without using transformer based models.

[gifs]



# How to use
## Installation
In order to run the program, you will need to install both `Docker` and `Python3.12` or greater, then `pip` and `poetry`. Then install the required dependencies by typing:
```bash
poetry install
poetry shell
```

## Running the program
Begin with editing `.env` file. Fill all the needed data - browser type and paths to files containing model weights.
Then activate your docker container with:
```bash
cd docker
sudo docker compose up
```
and run the program by typing:
```bash
python main.py
```


## Project Flow
The main flow of the project can be summarised as follows:
  - The user initiates the CAPTCHA-solving process by executing the `main.py` script.
  - The `CaptchaProcessor` class orchestrates the solution by:
    - Capturing a screenshot of the CAPTCHA using the `GuiAgent`.
    - Processing the image via the `ImageProcessor` generated by `ImageProcessorFactory` to cut it into pieces and extract relevant features.
    - Performing OCR on the extracted header image to identify the CAPTCHA type we are looking for.
    - Predicting the correct images or actions using pre-trained models (single or multi-image classifiers).
    - Handling mouse actions using the `MouseEngine`, which moves the mouse using chosen movement strategy.
  - Once the CAPTCHA is solved, the `CaptchaProcessor` handles the submission and proceeds to the next CAPTCHA if applicable.

## Directory Structure
The project was created as a mono-repository, and its structure is logically divided to ensure separation and maintainability. 


  - The [app/](/app/) directory holds the core operational files, including GUI interaction via `GuiAgent`, image processing tools, and mouse movement strategies.

  - The [gym/](/gym/) directory is dedicated to neural networks training. This includes training models for CAPTCHA solving and mouse movement, dataset preparation, and testing. By isolating training-related tasks in this module, the project maintains a clear distinction between training workflows and real-world operations.

  - The [docker/](/docker/) directory provides configurations for containerized deployment.


# How it all works?
## Screenshot and Segmentation
To ensure the CAPTCHA-solving program could process images despite various resolutions in various systems, a segmentation algorithm was implemented.
<div style="text-align: center;">
<img src="https://github.com/user-attachments/assets/d479d44d-e155-4cde-a0bc-b23736468e4f" width="90%" style="border-radius: 6px;">
</div>

## Machine Learning Models
Currently, breaking reCAPTCHA systems primarily involves solving two core challenges. Naturally, there are additional ones, but all can be addressed using models trained for these core tasks:
### Multiple Image CAPTCHA
<div style="text-align: center;">
<img src="https://github.com/user-attachments/assets/86b4601a-218a-4edb-9b6c-ab9805628408" width="30%" style="border-radius: 6px;">
</div>

For the $3 \times 3$ Multiple Objects CAPTCHA a transfer learning solution was used. The base architecture for the NN was ResNet-18, with its final fully-connected layer removed, thus transforming it into a feature extractor. The extracted features were then passed through a fully connected layer with 12 outputs, each representing the probability of the input belonging to a specific class.  

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/d2f4f7cb-bb72-46d9-9c00-0f5e0747d32f" width="20%" style="background-color: grey; padding:10%; border-radius: 10px;">
</div>
The model was trained in stages. First, fully frozen for few epochs, then with last layer unfrozen for few more epochs, and then with second-to-last layer unfrozen for the final training. The model was trained using the Adam optimizer with a varying learning rate.



### Single Image CAPTCHA

<div style="text-align: center;">
<img src="https://github.com/user-attachments/assets/298d010a-993e-4ea1-bf95-25ce2563867b" width="30%" style="border-radius: 6px;">
</div>

For the $4 \times 4$ Single Object CAPTCHA, again, a ResNet-18 structure was used, with its final fully connected layer removed. Then adjustments were done to address the challenge at hand - additional contextual information about the object class was added into the ResNet output by concatenating a class embedding with the extracted image features.

<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/d2f4f7cb-bb72-46d9-9c00-0f5e0747d32f" width="20%" style="background-color: grey; padding:10%; border-radius: 10px;">
</div>

The model was also trained in stages, with the same training strategy as the previous model. The model was trained using the Adam optimizer with a varying learning rate.


## Mouse Simulation
The mouse movement functionality was implemented using the Strategy design pattern, enabling seamless substitution of different movement algorithms.

One innovative strategy was a Generative Adversarial Network (GAN) for generating realistic mouse movements. To create the dataset for training, a simple game was developed where users clicked on a green square to start recording their mouse movements and a red square to stop. These recorded sequences were then used to train the GAN.

Training GANs, however, is notoriously challenging. Issues such as mode collapse and instability made it difficult to train the model robustly. Hence, a baseline deterministic algorithm was also implemented as a strategy.

