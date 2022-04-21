# The MiLe End Hums and Whistles Machine Learning Project

This year at Queen Mary University of London we are going to create a new dataset consisting of labelled audio recordings. Each
audio recording will consist of a unique interpretation of a small fragment of 8 iconic movie song.

We will consider fragments of approximately 15 seconds of duration from 8 songs.
The name of the songs, the label we will use to identify them (in parenthesis, bold font) and
an link to an online resource where you can listen to them are listed below:

- Harry Potter theme song (Potter)
https://youtu.be/Htaj3o3JD8I?t=0
- The Imperial March (StarWars)
https://youtu.be/s3SZ5sIMY6o?t=9
- Pink Panther theme song (Panther)
https://youtu.be/lp6z3s1Gig0?t=10
- Singing in the rain (Rain)
https://youtu.be/D1ZYhVpdXbQ?t=65
- Hakuna Matata (Hakuna)
https://youtu.be/MBIWFTXQbi4?t=79
- Mamma Mia (Mamma)
https://youtu.be/unfzfe8f9NI?t=50
- This is me (Showman)
https://youtu.be/CjxugyZCfuw?t=115
- Let it go (Frozen)
https://youtu.be/L0MK7qz13bU?t=126

---

### Data Interpretations
We will record two types of interpretations of the above mentioned songs:
- Humming.
- Whistling.

There is no right or wrong way of humming or whistling to a song. When recording ourself, we just hum or whistle as you would normally do (da-da-da, la-la-la, hm-hm-hm, ti-ro-ri, pa-rapa…). We did not sing the lyrics.

---

### Jupyter Notebooks
**Basic solution :**
> Using the MLEnd Hums and Whistles dataset, build a machine learning pipeline that takes as an input a Potter or a StarWars audio segment and predicts its song label (either Harry or StarWars).

**Underline Steps:**
- Importing required python libraries
- Data Cleaning Function
- Reading and processing Harry Potter and Starwars audio files
- Merging and creating final dataframe
- Feature Extraction from the audio: Power, Pitch Mean, Pitch Std., Voice Frame, Interpretation Label, Song Label
- Data Exploration, Data Normalization, Data Split
- Dummy check for Humming and whistling classification.
- Model 1: SVM classifier for classifying Harry Potter or Starwars files
- Analysing the results:

  ![image](https://user-images.githubusercontent.com/25953832/164483306-33af3861-cb22-4263-ba66-43ca32d164bf.png)

   - Training Accuracy: 0.6840277777777778
   - Validation Accuracy: 0.5874439461883408
   - Testing Accuracy:0.56 
   
We can improve the model by including advance features of adio processing like ```mfcc, chroma, melody (to be included in the advance solution)```

Advanced solution : 
> An advanced Machine Learning solution to identify different audio files 

**Underline Steps:**
- Data Processing of 7 songs
- Feature Extractions: Previously we used following features from the audio data:
> Power, Pitch Mean, Pitch Std., Voice Frame, Interpretation Label, Song Label
- Advance features which we have added are:
>  MFFC, Chroma, Mel-fre, Contrast
- Feature scaling using z-scoreS
- Model 1: Modified SVM Model
> - Training Accuracy 0.5252725470763132
> - Validation  Accuracy 0.38125802310654683
> - Testing Accuracy 0.39080459770114945
- Model 2: CNN
> - Training Accuracy:  0.9856293201446533
> - Validation Accuracy:  0.43132221698760986
> - Testing Accuracy 0.41379310344827586

- Unsupervised Gender Classification using hierarcial clustering based on mfcc feature of the audio files.
SVM Model:
