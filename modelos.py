import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# carregando os dados
def dados():
    df = pd.read_csv('ames.csv')
    return df

# engenharia de features
def engenharia(df):
    # novas
    """
    TotalSF: Área total combinando porão e andares
    TotalBathrooms: Número total de banheiros
    TotalPorchSF: Área total de varandas
    Age: Idade do imóvel
    IsNew: Indicador para imóveis novos
    HasPool: Indicador de piscina
    """
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['Age'] = df['YrSold'] - df['YearBuilt']
    df['IsNew'] = df['Age'].apply(lambda x: 1 if x <= 2 else 0)
    
    return df

# preparação dos dados
def preparacao(df):
    # separar features numéricas e categóricas
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_features = numeric_features.drop('SalePrice')
    categorical_features = df.select_dtypes(include=['object']).columns
    
    # criar preprocessadores
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse=False))
    ])
    
    # combinar eles
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# definir os modelos
def cria_modelos():
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    param_grids = {
        'Ridge': {'regressor__alpha': [0.1, 1.0, 10.0]},
        'Lasso': {'regressor__alpha': [0.1, 1.0, 10.0]},
        'Random Forest': {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [10, 20, None]
        },
        'Gradient Boosting': {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.01, 0.1]
        }
    }
    
    return models, param_grids

# treinando e avaliando
def treina(X_train, X_test, y_train, y_test, preprocessor, models, param_grids):
    results = {}
    best_models = {}
    
    for name, model in models.items():
        print(f"\nTreinando {name}...")
        
        # pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # grid search se houver parâmetros para ajustar
        if name in param_grids:
            grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_models[name] = grid_search.best_estimator_
            y_pred = grid_search.predict(X_test)
            print(f"Melhores parâmetros: {grid_search.best_params_}")
        else:
            pipeline.fit(X_train, y_train)
            best_models[name] = pipeline
            y_pred = pipeline.predict(X_test)
        
        # cálculo das métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.4f}")
    
    return results, best_models

# análise
def feature_importance(best_model, feature_names):
    if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
        importance = best_model.named_steps['regressor'].feature_importances_
        feat_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        return feat_importance.sort_values('Importance', ascending=False).head(10)
    return None

def main():
    df = dados()
    df = engenharia(df)
    
    # separa features e target
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = preparacao(df)
    
    models, param_grids = cria_modelos()
    results, best_models = treina(X_train, X_test, y_train, y_test, preprocessor, models, param_grids)
    
    # analisa feature importance para o Random Forest
    if 'Random Forest' in best_models:
        feature_names = (X_train.columns.tolist() +
                        list(best_models['Random Forest']
                            .named_steps['preprocessor']
                            .named_transformers_['cat']
                            .named_steps['onehot']
                            .get_feature_names(X_train.select_dtypes(include=['object']).columns)))
        importance_df = feature_importance(best_models['Random Forest'], feature_names)
        if importance_df is not None:
            print("\nTop 10 Features mais importantes:")
            print(importance_df)
    
    return results, best_models

if __name__ == "__main__":
    results, best_models = main()