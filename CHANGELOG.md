# WHAT'S NEW

# 0.13.0
- Added new parameter 'select_models' that enables you to select only a few models to train with instead of using all models at once.
- Added progress bar when training models
- Added more understandable error message with fixes indicated
- Fixed key error bug in use_model when specifying metric
- Removed r2 score metric from classification

# 0.12.3
- Fixed bug that stopped models from training due to inconsistent number of columns
- Temporarily disabled using over, under or over_under sampling techniques when using kf=True
- Added a new parameter y to 'visualize' and 'show' methods to indicate the target.

# 0.12.0 
- If model is unable to properly compute metrics, it's value is replaced with np.nan
- Added 'encode' parameter in split method to encode categorical columns
- Added 'missing' values parameter in split method for filling missing values for both numerical and categorical columns.
# 0.11.21 - BUG FIX
- Added missing requirement for kaleido engine
# 0.11.0 - PATCH RELEASE
- Removed the target_class parameter in the instance of the MultiClass object, the class of your target is automatically checked.
# 0.1.31
- Added support for dimensionality reduction in split method in MultiRegressor
