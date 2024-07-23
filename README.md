This project involves the development of an AI-based music critique system, designed to rate newly produced songs or tracks on a scale of 1.0 to 5.0. The system leverages a feedforward neural network with a single output node. The training dataset comprises the top 1000 songs of all time, as well as a comprehensive dataset of 1.4 million songs from Spotify.

**Data Collection and Feature Extraction**: 
- **Top 1000 Songs**: Audio features were extracted using the Librosa library.
- **Spotify Songs**: Features were obtained via Spotify's Open API. The features include spectral features, chroma features, chromagram, chords, tonnetz, MFCCs (Mel-Frequency Cepstral Coefficients), and more.

**Feature Engineering**:
- **Spectral Features**: Such as spectral centroid, bandwidth, contrast, and roll-off.
- **Chroma Features**: Including chroma energy and chroma STFT.
- **Rhythm Features**: Tempo, beat, and onset strength.
- **Harmonic Features**: Harmonic-to-noise ratio.
- **Timbre Features**: MFCCs and chroma-based timbre features.
- **Popularity**: Used as the target variable for the Spotify dataset.

**Model Training**:
- The feedforward neural network was trained using these features to predict song ratings.
- Extensive feature analysis was conducted to select the most impactful features, using techniques such as correlation matrices, feature importance scores, and principal component analysis (PCA).
- Data visualization through multiple graphs helped in understanding feature distributions and relationships.

**Deployment**:
- The system was deployed on AWS EC2 instances, ensuring scalability and reliability.
- Songs are uploaded to an S3 bucket, from which the system retrieves them for processing.
- The trained model generates ratings for the songs, which are then outputted.

**Integration with OpenAI**:
- OpenAI's API was integrated to provide explanations for the ratings based on feature values. This added layer of interpretability helps users understand the rationale behind each rating.

**Performance**:
- The model achieved an accuracy of 87% in rating songs.
- Techniques such as cross-validation, hyperparameter tuning, and regularization were employed to optimize model performance.

**Outcome**:
- The model provides a reliable and automated means to rate new music tracks, aiding artists and producers in evaluating their work before market release.

This comprehensive system integrates advanced audio signal processing techniques, robust machine learning algorithms, extensive data analysis, and scalable cloud deployment, showcasing a practical application of AI in the music industry.
