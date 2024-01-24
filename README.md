# Sales_Dynamite: Retail Sales Forecasting System

## Introduction

Sales_Dynamite is a platform specifically developed to address the complexities of sales prediction and trend analysis in the retail sector. This system integrates the advanced capabilities of XGBoost and Prophet, two leading machine learning models, to deliver precise and reliable sales revenue forecasts. It is meticulously deployed using FastAPI on the Heroku platform, ensuring a seamless and scalable user experience. This makes Sales_Dynamite an invaluable tool for businesses seeking to optimize their inventory management strategies and refine their financial planning processes. With its user-friendly interface, Sales_Dynamite empowers retailers with actionable insights, fostering better decision-making for sustainable growth and competitive advantage in the market.


## Repository Structure

Sales_Dynamite/

    ├── docs               
    │
    ├── models/            <- Trained and serialized models, model predictions, or model summaries
    │   ├── forecasting/
    │   │   └── forecasting_model.joblib
    │   ├── predictive/
    │   │   └── regression_model.joblib
    │   ├── cat_id_encoder.joblib
    │   ├── dept_id_encoder.joblib
    │   ├── event_name_encoder.joblib
    │   ├── event_type_encoder.joblib
    │   ├── item_id_encoder.joblib
    │   ├── state_id_encoder.joblib
    │   ├── store_id_encoder.joblib          
    │
    ├── notebooks/           <- Jupyter notebooks. 
    │   ├── forecasting/
    │   │   └── sharma_shivatmak-14233934-forecasting_phophet.ipynb
    │   └── predictive/
    │       └── sharma_shivatmak-14233934-predicitive_xgboost.ipynb          
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src/               <- source code for use in this project
    │    ├── data/
    │    │   ├── data_loader.py
    │    │   └── data_loader_f.py
    │    ├── features/
    │    │   └── data_transformer.py
    │        └── data_transformer_f.py
    │    ├── models/
    │    │   ├── forecasting_model.py
    │    │   └── predict_model.py
    │
    ├── LICENSE
    ├── Makefile          
    ├── Procfile
    ├── README.md          <- The top-level README for developers using this project.
    ├── main_app.py
    ├── requirements.txt
    ├── runtime.txt
    ├── setup.py
    ├── test_environment.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## Repository Setup:

-> Clone the repository.

-> Navigate to the project's root directory.

-> Install dependencies: ``` pip install -r requirements.txt ```

-> Deploying on Heroku

## Preparation:

Ensure you have a Heroku account and the Heroku CLI.

## Deployment Steps:

-> Login to Heroku: ```heroku login```

-> Create a Heroku app: ```heroku create [app-name]```

-> Set Heroku as a remote: ```git remote add heroku [heroku-git-url]```

-> Deploy: ```git push heroku master```

-> Open the app: ```heroku open```


## Licensing

Sales_Dynamite is made available under the MIT license.
