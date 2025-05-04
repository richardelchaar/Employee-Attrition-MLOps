# Frontend Application

This directory contains the Streamlit-based frontend application for the Employee Attrition prediction system. The frontend provides an interactive interface for users to make predictions and view model information.

## Components

### Main Application (`app.py`)
- Streamlit application entry point
- Page routing and navigation
- Main layout and styling

### Pages
- **Home Page**: Overview and quick start
- **Prediction Page**: Interactive prediction interface
- **Model Info Page**: Model performance and metrics
- **Drift Monitoring Page**: Drift detection results
- **Documentation Page**: User guides and API docs

### Components
- **Input Forms**: Data entry components
- **Visualizations**: Charts and graphs
- **Results Display**: Prediction results
- **Navigation**: Page navigation

## Features

### Prediction Interface
- Interactive form for feature input
- Real-time prediction display
- Confidence score visualization
- Feature importance explanation

### Model Information
- Performance metrics display
- Training history
- Feature importance charts
- Model version information

### Drift Monitoring
- Drift detection results
- Feature drift visualization
- Prediction drift charts
- Alert display

## Implementation

### Running the Application
```bash
streamlit run src/frontend/app.py
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export API_URL=http://localhost:8000
```

## Usage

### Making Predictions
1. Navigate to the Prediction page
2. Fill in employee information
3. Click "Predict" button
4. View prediction results and explanations

### Viewing Model Information
1. Navigate to the Model Info page
2. Select model version
3. View performance metrics
4. Explore feature importance

### Monitoring Drift
1. Navigate to the Drift Monitoring page
2. View current drift status
3. Explore drift visualizations
4. Check alert history

## Configuration

### Environment Variables
```env
STREAMLIT_SERVER_PORT=8501
API_URL=http://localhost:8000
MODEL_VERSION=latest
```

### Theme Configuration
```python
theme = {
    "primaryColor": "#FF4B4B",
    "backgroundColor": "#FFFFFF",
    "secondaryBackgroundColor": "#F0F2F6",
    "textColor": "#262730",
    "font": "sans serif"
}
```

## Best Practices

1. **User Experience**
   - Clear navigation
   - Intuitive forms
   - Responsive design
   - Helpful error messages

2. **Performance**
   - Optimize loading times
   - Cache predictions
   - Efficient data fetching
   - Smooth transitions

3. **Maintenance**
   - Regular updates
   - Bug fixes
   - Feature additions
   - Documentation updates

4. **Security**
   - Input validation
   - Error handling
   - Secure API calls
   - Data protection

## Testing

Run the frontend test suite:
```bash
pytest tests/test_frontend.py
```

## Development

### Adding New Features
1. Create new page component
2. Add to navigation
3. Implement functionality
4. Add tests
5. Update documentation

### Styling Guidelines
- Use consistent color scheme
- Follow Streamlit best practices
- Maintain responsive design
- Ensure accessibility

## Documentation

### User Guide
- Installation instructions
- Usage examples
- Troubleshooting
- FAQ

### API Documentation
- Endpoint descriptions
- Request/response formats
- Authentication
- Error handling

## Deployment

### Docker Deployment
```bash
docker build -t frontend -f Dockerfile.frontend .
docker run -p 8501:8501 frontend
```

### Environment Variables
```env
STREAMLIT_SERVER_PORT=8501
API_URL=http://api:8000
MODEL_VERSION=latest
``` 