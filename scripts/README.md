# Utility Scripts

This directory contains utility scripts and automation tools for the Employee Attrition MLOps project. These scripts help with various tasks including data processing, model management, and system maintenance.

## Script Categories

### Data Management
- Data preprocessing scripts
- Reference data generation
- Data validation tools
- Data export utilities

### Model Management
- Model training scripts
- Model evaluation tools
- Model deployment scripts
- Model versioning utilities

### System Maintenance
- Database maintenance
- Log management
- System cleanup
- Backup utilities

### Automation
- Scheduled tasks
- Batch processing
- Monitoring scripts
- Alert management

## Key Scripts

### Data Processing
```bash
# Preprocess data
python scripts/preprocess_data.py --input data/raw --output data/processed

# Generate reference data
python scripts/generate_reference_data.py --data data/processed --output references/

# Validate data
python scripts/validate_data.py --data data/processed --schema schemas/data_schema.json
```

### Model Management
```bash
# Train model
python scripts/train_model.py --config configs/model_config.yaml

# Evaluate model
python scripts/evaluate_model.py --model models/model.pkl --data data/test.csv

# Deploy model
python scripts/deploy_model.py --model models/model.pkl --version 1.0.0
```

### System Maintenance
```bash
# Clean up old models
python scripts/cleanup_models.py --older-than 30d

# Backup database
python scripts/backup_database.py --output backups/

# Rotate logs
python scripts/rotate_logs.py --keep 7
```

## Usage Examples

### Data Processing
```python
# Example: Preprocess data
from scripts.preprocess_data import preprocess_data

# Process data
processed_data = preprocess_data(
    input_path="data/raw",
    output_path="data/processed",
    config="configs/preprocessing.yaml"
)
```

### Model Management
```python
# Example: Train model
from scripts.train_model import train_model

# Train model
model = train_model(
    data_path="data/train.csv",
    config_path="configs/model_config.yaml",
    output_path="models/"
)
```

### System Maintenance
```python
# Example: Cleanup
from scripts.cleanup import cleanup_old_files

# Cleanup old files
cleanup_old_files(
    directory="models/",
    pattern="*.pkl",
    days_old=30
)
```

## Configuration

### Script Configuration
```yaml
# configs/scripts_config.yaml
data_processing:
  input_dir: data/raw
  output_dir: data/processed
  validation: true

model_management:
  model_dir: models/
  version_format: "{major}.{minor}.{patch}"
  backup: true

maintenance:
  cleanup_days: 30
  backup_frequency: daily
  log_rotation: weekly
```

### Environment Variables
```env
# .env
DATA_DIR=data/
MODEL_DIR=models/
BACKUP_DIR=backups/
LOG_DIR=logs/
```

## Best Practices

1. **Script Design**
   - Clear purpose
   - Modular structure
   - Error handling
   - Logging

2. **Implementation**
   - Use configuration files
   - Handle edge cases
   - Clean up resources
   - Validate inputs

3. **Maintenance**
   - Regular updates
   - Documentation
   - Testing
   - Version control

4. **Documentation**
   - Usage instructions
   - Parameters
   - Examples
   - Dependencies

## Automation

### Scheduled Tasks
```bash
# Example crontab entry
0 0 * * * python scripts/backup_database.py
0 1 * * * python scripts/cleanup_models.py
0 2 * * * python scripts/rotate_logs.py
```

### Batch Processing
```bash
# Process multiple files
python scripts/batch_process.py --input-dir data/raw --output-dir data/processed

# Train multiple models
python scripts/batch_train.py --config-dir configs/models/
```

## Testing

### Script Testing
```bash
# Run script tests
pytest tests/test_scripts.py

# Test specific script
pytest tests/test_scripts.py::test_preprocess_data
```

### Integration Testing
```bash
# Test script integration
python scripts/test_integration.py --config configs/integration_test.yaml
```

## Troubleshooting

### Common Issues
1. **Script Failures**
   - Check logs
   - Verify inputs
   - Review configuration
   - Check permissions

2. **Performance Issues**
   - Optimize code
   - Use caching
   - Batch processing
   - Resource management

3. **Integration Issues**
   - Verify connections
   - Check dependencies
   - Review configurations
   - Test environment

## Maintenance

### Regular Tasks
- Update scripts
- Review configurations
- Test functionality
- Update documentation

### Version Control
- Script versioning
- Configuration versioning
- Documentation versioning
- Test versioning 