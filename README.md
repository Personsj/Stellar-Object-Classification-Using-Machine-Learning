Stellar Object Classification Using Machine Learning

Background and Functionality:
This project utilizes Python programming to analyze the Stellar Classification Dataset - SDSS17 dataset from the Sloan Digital Sky Survey, performing data loading, feature selection, and standardization. Afterwards, scikit-learn was employed to create and evaluate a RandomForestClassifier model, and then the importance of dataset features were ranked for the classification of stellar entities as stars, galaxies, or quasars. Lastly, a deep neural network was implemented, trained, and tested using PyTorch for further evaluation.

Dataset Features:
- obj_ID — Object ID, or a unique identifier for each object
- alpha — used to denote the right ascension angle of the object (at J2000 epoch)
- delta — used to denote the declination angle (at J2000 epoch)
- u — apparent brightness in the ultraviolet filter
- g — apparent brightness in the green filter
- r — apparent brightness in the red filter
- i — apparent brightness in the near-infrared filter
- z — apparent brightness in the far-infrared filter
- run_ID — run number to identify the specific scan
- rerun_ID — rerun number to specify how the image was processed
- cam_col — camera column that specifies the scanline within the run
- field_ID — field ID to specify the field
- spec_obj_ID — unique spectroscopic object ID (meaning that 2 different with the same spectroscopic object ID must share the same output class)
- class — specifies the classification of the object as GALAXY, STAR, or QSO.
- redshift — redshift value (recessional velocity of the object) based on wavelength increase
- plate — identifies plate number in SDSS system
- MJD — Modified Julian Date indicating when the piece of SDSS data was taken
- fiber_ID — specifies the fiber that pointed light at the focal plane in each observation
