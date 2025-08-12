# ============================================================================
# PHASE 2: STATISTICAL ANALYSIS ENGINE
# ============================================================================

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalysisEngine:
    """
    Core engine for statistical analysis and feature discrimination
    This is the brain of your AI Agent!
    """
    
    def __init__(self, feature_matrix_path):
        """Initialize with your feature matrix CSV"""
        print("🚀 Initializing Statistical Analysis Engine...")
        
        # Load your feature matrix
        self.feature_matrix = pd.read_csv(feature_matrix_path)
        print(f"✅ Loaded feature matrix: {self.feature_matrix.shape}")
        
        # Initialize storage for results
        self.class_statistics = {}
        self.discriminative_features = []
        self.model_results = {}
        self.insights = []
        
        # Display basic info
        self._display_dataset_info()
    
    def _display_dataset_info(self):
        """Show basic information about the dataset"""
        print("\n📊 DATASET INFORMATION:")
        print(f"Total samples: {len(self.feature_matrix)}")
        print(f"Total features: {len(self.feature_matrix.columns) - 2}")  # exclude label, sample
        
        # Show class distribution
        print(f"\nClass distribution:")
        class_counts = self.feature_matrix['label'].value_counts()
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} samples")
        
        # Show sample features
        feature_cols = [col for col in self.feature_matrix.columns if col not in ['label', 'sample']]
        print(f"\nExample features: {feature_cols[:5]}...")
    
    def calculate_class_statistics(self, binary_mode=True):
        """
        Step 2.1: Calculate comprehensive statistics for each class
        This identifies which features differ most between OK and KO
        """
        print("\n🔍 STEP 2.1: Calculating Class-wise Statistics...")
        
        # Prepare data
        X = self.feature_matrix.drop(['label', 'sample'], axis=1, errors='ignore')
        y = self.feature_matrix['label']
        
        # Convert to binary classification if requested
        if binary_mode:
            y_processed = y.apply(lambda x: 'OK' if x == 'OK' else 'KO')
            print("Using binary classification: OK vs KO")
        else:
            y_processed = y
            print("Using multi-class classification")
        
        # Clean data
        X_clean = self._clean_features(X)
        
        # Calculate statistics for each feature
        feature_stats = {}
        
        for feature in X_clean.columns:
            try:
                feature_data = X_clean[feature]
                
                # Skip if all NaN or constant
                if feature_data.isna().all() or feature_data.nunique() <= 1:
                    continue
                
                # Get data for each class
                class_data = {}
                for class_name in y_processed.unique():
                    class_mask = y_processed == class_name
                    class_values = feature_data[class_mask].dropna()
                    
                    if len(class_values) == 0:
                        continue
                    
                    class_data[class_name] = {
                        'values': class_values,
                        'mean': class_values.mean(),
                        'std': class_values.std(),
                        'median': class_values.median(),
                        'min': class_values.min(),
                        'max': class_values.max(),
                        'count': len(class_values)
                    }
                
                # Skip if we don't have data for at least 2 classes
                if len(class_data) < 2:
                    continue
                
                # Calculate discrimination metrics
                discrimination_metrics = self._calculate_discrimination_metrics(class_data, feature)
                
                feature_stats[feature] = {
                    'class_statistics': class_data,
                    'discrimination_metrics': discrimination_metrics
                }
                
            except Exception as e:
                print(f"⚠️  Warning: Could not process feature {feature}: {e}")
                continue
        
        # Sort features by discrimination power
        sorted_features = sorted(feature_stats.items(), 
                               key=lambda x: x[1]['discrimination_metrics']['separation_score'], 
                               reverse=True)
        
        self.class_statistics = dict(sorted_features)
        
        print(f"Calculated statistics for {len(self.class_statistics)} features")
        print(f"Top 3 most discriminative features:")
        
        for i, (feature, stats) in enumerate(list(self.class_statistics.items())[:3], 1):
            score = stats['discrimination_metrics']['separation_score']
            print(f"   {i}. {feature}: separation_score = {score:.4f}")
        
        return self.class_statistics
    
    def _clean_features(self, X):
        """Clean feature data"""
        print("Cleaning feature data...")
        
        # Convert to numeric
        X_clean = X.apply(pd.to_numeric, errors='coerce')
        
        # Handle infinite values
        inf_count = np.isinf(X_clean.values).sum()
        if inf_count > 0:
            print(f"   Replacing {inf_count} infinite values with NaN")
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Handle very large numbers
        threshold = 1e10
        large_count = (np.abs(X_clean) > threshold).sum().sum()
        if large_count > 0:
            print(f"   Replacing {large_count} very large values (>{threshold}) with NaN")
        X_clean = X_clean.mask(np.abs(X_clean) > threshold, np.nan)
        
        # Fill missing values with median
        missing_before = X_clean.isnull().sum().sum()
        X_clean = X_clean.fillna(X_clean.median())
        X_clean = X_clean.fillna(0)  # If median is also NaN
        
        if missing_before > 0:
            print(f"   Filled {missing_before} missing values with median")
        
        return X_clean
    
    def _calculate_discrimination_metrics(self, class_data, feature_name):
        """Calculate how well a feature discriminates between classes"""
        
        # For binary classification (OK vs KO)
        if len(class_data) == 2:
            classes = list(class_data.keys())
            class1_data = class_data[classes[0]]['values']
            class2_data = class_data[classes[1]]['values']
            
            # Statistical significance test
            try:
                t_stat, p_value = stats.ttest_ind(class1_data, class2_data)
                mannwhitney_stat, mannwhitney_p = stats.mannwhitneyu(class1_data, class2_data, alternative='two-sided')
            except:
                t_stat, p_value = 0, 1
                mannwhitney_stat, mannwhitney_p = 0, 1
            
            # Effect size (Cohen's d)
            mean1, mean2 = class1_data.mean(), class2_data.mean()
            std1, std2 = class1_data.std(), class2_data.std()
            
            # Pooled standard deviation
            n1, n2 = len(class1_data), len(class2_data)
            pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
            
            cohens_d = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            
            # Custom separation score (combines effect size and significance)
            separation_score = cohens_d * (1 - p_value) if p_value < 1 else 0
            
            # Interpretation
            effect_interpretation = self._interpret_effect_size(cohens_d)
            significance_interpretation = "Significant" if p_value < 0.05 else "Not significant"
            
            # Mean difference and direction
            mean_difference = abs(mean1 - mean2)
            higher_class = classes[0] if mean1 > mean2 else classes[1]
            
            return {
                'separation_score': separation_score,
                'cohens_d': cohens_d,
                'effect_size_interpretation': effect_interpretation,
                'p_value': p_value,
                'statistical_significance': significance_interpretation,
                'mean_difference': mean_difference,
                'higher_mean_class': higher_class,
                't_statistic': abs(t_stat),
                'mannwhitney_p': mannwhitney_p,
                'classes_compared': classes
            }
        
        else:
            # Multi-class discrimination (simplified)
            all_values = []
            class_means = []
            
            for class_name, data in class_data.items():
                all_values.extend(data['values'].tolist())
                class_means.append(data['mean'])
            
            # ANOVA F-statistic
            try:
                class_values = [data['values'].tolist() for data in class_data.values()]
                f_stat, anova_p = stats.f_oneway(*class_values)
            except:
                f_stat, anova_p = 0, 1
            
            # Variance between classes vs within classes
            between_class_var = np.var(class_means)
            total_var = np.var(all_values)
            separation_score = (between_class_var / total_var) * (1 - anova_p) if total_var > 0 else 0
            
            return {
                'separation_score': separation_score,
                'f_statistic': f_stat,
                'anova_p_value': anova_p,
                'statistical_significance': "Significant" if anova_p < 0.05 else "Not significant",
                'between_class_variance': between_class_var,
                'total_variance': total_var
            }
    
    def _interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size"""
        if cohens_d < 0.2:
            return "Small effect"
        elif cohens_d < 0.5:
            return "Small to medium effect"
        elif cohens_d < 0.8:
            return "Medium to large effect"
        else:
            return "Large effect"
    
    def get_best_discriminative_features(self, top_n=10, min_separation_score=0.01):
        """
        Step 2.2: Get the top N features that best discriminate between classes
        This is what the GUI will display as "best statistical indices"
        """
        print(f"\n🏆 STEP 2.2: Getting Top {top_n} Discriminative Features...")
        
        if not self.class_statistics:
            print("⚠️  No class statistics available. Running analysis first...")
            self.calculate_class_statistics()
        
        # Filter features by minimum separation score
        filtered_features = {
            feature: stats for feature, stats in self.class_statistics.items()
            if stats['discrimination_metrics']['separation_score'] >= min_separation_score
        }
        
        print(f"📊 Found {len(filtered_features)} features with separation_score >= {min_separation_score}")
        
        # Get top N features
        top_features = list(filtered_features.items())[:top_n]
        
        # Format results for GUI display
        discriminative_features = []
        
        for i, (feature, stats) in enumerate(top_features, 1):
            class_stats = stats['class_statistics']
            disc_metrics = stats['discrimination_metrics']
            
            # Extract sensor information
            sensor_parts = feature.split('_')
            sensor_name = sensor_parts[0] if len(sensor_parts) > 0 else "Unknown"
            channel = '_'.join(sensor_parts[1:-1]) if len(sensor_parts) > 2 else "Unknown"
            statistic = sensor_parts[-1] if len(sensor_parts) > 0 else "Unknown"
            
            # Create feature summary
            feature_info = {
                'rank': i,
                'feature_name': feature,
                'sensor_name': sensor_name,
                'channel': channel,
                'statistic_type': statistic,
                'separation_score': round(disc_metrics['separation_score'], 4),
                'effect_size': round(disc_metrics.get('cohens_d', 0), 4),
                'effect_interpretation': disc_metrics.get('effect_size_interpretation', 'N/A'),
                'statistical_significance': disc_metrics['statistical_significance'],
                'p_value': round(disc_metrics.get('p_value', 1), 6)
            }
            
            # Add class-specific information
            if 'classes_compared' in disc_metrics:
                classes = disc_metrics['classes_compared']
                feature_info['classes_compared'] = classes
                
                # Add mean values for each class
                for class_name in classes:
                    if class_name in class_stats:
                        feature_info[f'{class_name}_mean'] = round(class_stats[class_name]['mean'], 4)
                        feature_info[f'{class_name}_std'] = round(class_stats[class_name]['std'], 4)
                
                # Add interpretation
                if len(classes) == 2:
                    higher_class = disc_metrics['higher_mean_class']
                    mean_diff = disc_metrics['mean_difference']
                    feature_info['interpretation'] = f"{higher_class} has higher values (diff: {mean_diff:.4f})"
            
            discriminative_features.append(feature_info)
        
        self.discriminative_features = discriminative_features
        
        # Display results
        print(f"✅ Top {len(discriminative_features)} discriminative features:")
        print(f"{'Rank':<4} {'Feature':<40} {'Sensor':<15} {'Sep.Score':<10} {'Significance':<15}")
        print("-" * 90)
        
        for feature in discriminative_features[:5]:  # Show top 5
            print(f"{feature['rank']:<4} {feature['feature_name'][:38]:<40} {feature['sensor_name']:<15} "
                  f"{feature['separation_score']:<10} {feature['statistical_significance']:<15}")
        
        if len(discriminative_features) > 5:
            print(f"... and {len(discriminative_features) - 5} more features")
        
        return discriminative_features
    
    def train_discrimination_models(self):
        """
        Step 2.3: Train ML models to validate feature importance
        This provides an additional perspective on discriminative features
        """
        print(f"\n🤖 STEP 2.3: Training ML Models for Feature Validation...")
        
        # Prepare data
        X = self.feature_matrix.drop(['label', 'sample'], axis=1, errors='ignore')
        y = self.feature_matrix['label'].apply(lambda x: 'OK' if x == 'OK' else 'KO')
        
        # Clean data
        X_clean = self._clean_features(X)
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
            )
        except ValueError:
            # If stratification fails (very small dataset), use simple split
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_encoded, test_size=0.3, random_state=42
            )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train different models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        model_results = {}
        
        for name, model in models.items():
            try:
                print(f"   Training {name}...")
                
                # Train model
                if name == 'Logistic Regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_[0])
                else:
                    importance = np.ones(X_clean.shape[1])
                
                # Create feature importance dataframe
                feature_importance = pd.DataFrame({
                    'feature': X_clean.columns,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                model_results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'feature_importance': feature_importance.head(10).to_dict('records'),
                    'scaler': scaler if name == 'Logistic Regression' else None,
                    'label_encoder': le
                }
                
                print(f"   ✅ {name}: Accuracy = {accuracy:.4f}")
                
            except Exception as e:
                print(f"   ❌ Failed to train {name}: {e}")
                continue
        
        self.model_results = model_results
        
        # Display top features from each model
        print(f"\n📊 Top 5 important features by model:")
        for model_name, results in model_results.items():
            print(f"\n{model_name}:")
            for i, feature_info in enumerate(results['feature_importance'][:5], 1):
                print(f"   {i}. {feature_info['feature']}: {feature_info['importance']:.4f}")
        
        return model_results
    
    def generate_insights(self):
        """
        Generate human-readable insights about the analysis
        This is what will be displayed in the GUI
        """
        print(f"\n💡 GENERATING INSIGHTS...")
        
        insights = []
        
        # Dataset insights
        total_samples = len(self.feature_matrix)
        ok_samples = len(self.feature_matrix[self.feature_matrix['label'] == 'OK'])
        ko_samples = total_samples - ok_samples
        
        insights.append(f"Dataset contains {total_samples} samples: {ok_samples} OK and {ko_samples} KO samples")
        
        # Feature discrimination insights
        if self.discriminative_features:
            top_feature = self.discriminative_features[0]
            insights.append(f"Most discriminative feature: '{top_feature['feature_name']}' from {top_feature['sensor_name']} sensor")
            insights.append(f"This feature has a {top_feature['effect_interpretation'].lower()} with separation score of {top_feature['separation_score']}")
            
            # Count significant features
            significant_features = [f for f in self.discriminative_features if f['statistical_significance'] == 'Significant']
            insights.append(f"Found {len(significant_features)} statistically significant discriminative features")
            
            # Sensor analysis
            sensor_counts = {}
            for feature in self.discriminative_features[:10]:  # Top 10
                sensor = feature['sensor_name']
                sensor_counts[sensor] = sensor_counts.get(sensor, 0) + 1
            
            if sensor_counts:
                most_important_sensor = max(sensor_counts, key=sensor_counts.get)
                insights.append(f"Sensor '{most_important_sensor}' has the most discriminative features ({sensor_counts[most_important_sensor]} in top 10)")
        
        # Model performance insights
        if self.model_results:
            best_model = max(self.model_results.items(), key=lambda x: x[1]['accuracy'])
            insights.append(f"Best ML model: {best_model[0]} with {best_model[1]['accuracy']:.1%} accuracy")
        
        self.insights = insights
        
        print("✅ Generated insights:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
        
        return insights
    
    def get_analysis_summary(self):
        """
        Get complete analysis summary for GUI display
        This is the main function your teammate will call
        """
        print(f"\n📋 GENERATING COMPLETE ANALYSIS SUMMARY...")
        
        # Run analysis if not done
        if not self.class_statistics:
            self.calculate_class_statistics()
        
        if not self.discriminative_features:
            self.get_best_discriminative_features()
        
        if not self.model_results:
            self.train_discrimination_models()
        
        if not self.insights:
            self.generate_insights()
        
        # Create comprehensive summary
        summary = {
            'dataset_info': {
                'total_samples': len(self.feature_matrix),
                'total_features': len(self.feature_matrix.columns) - 2,
                'class_distribution': self.feature_matrix['label'].value_counts().to_dict(),
                'ok_samples': len(self.feature_matrix[self.feature_matrix['label'] == 'OK']),
                'ko_samples': len(self.feature_matrix[self.feature_matrix['label'] != 'OK'])
            },
            'best_discriminative_features': self.discriminative_features,
            'model_performance': {
                name: {
                    'accuracy': results['accuracy'],
                    'top_features': results['feature_importance'][:5]
                } for name, results in self.model_results.items()
            },
            'statistical_insights': self.insights,
            'analysis_metadata': {
                'features_analyzed': len(self.class_statistics),
                'significant_features': len([f for f in self.discriminative_features 
                                           if f['statistical_significance'] == 'Significant']),
                'top_sensors': self._get_top_sensors()
            }
        }
        
        print("✅ Analysis summary ready for GUI!")
        return summary
    
    def _get_top_sensors(self):
        """Get sensors with most discriminative features"""
        if not self.discriminative_features:
            return {}
        
        sensor_counts = {}
        for feature in self.discriminative_features[:20]:  # Top 20
            sensor = feature['sensor_name']
            sensor_counts[sensor] = sensor_counts.get(sensor, 0) + 1
        
        return dict(sorted(sensor_counts.items(), key=lambda x: x[1], reverse=True))

# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

def test_statistical_engine():
    """Test the complete statistical analysis engine"""
    print("TESTING STATISTICAL ANALYSIS ENGINE")
    print("=" * 60)
    
    # Initialize engine (replace with your CSV path)
    engine = StatisticalAnalysisEngine("feature_matrix.csv")
    
    # Run complete analysis
    summary = engine.get_analysis_summary()
    
    # Display key results
    print(f"\n KEY RESULTS:")
    print(f"Dataset: {summary['dataset_info']['total_samples']} samples, {summary['dataset_info']['total_features']} features")
    
    print(f"\n Top 3 Discriminative Features:")
    for i, feature in enumerate(summary['best_discriminative_features'][:3], 1):
        print(f"   {i}. {feature['feature_name']}")
        print(f"      Sensor: {feature['sensor_name']}")
        print(f"      Separation Score: {feature['separation_score']}")
        print(f"      Significance: {feature['statistical_significance']}")
        if 'OK_mean' in feature and 'KO_mean' in feature:
            print(f"      OK mean: {feature['OK_mean']}, KO mean: {feature['KO_mean']}")
        print()
    
    print(f" Model Performance:")
    for model, perf in summary['model_performance'].items():
        print(f"   {model}: {perf['accuracy']:.1%} accuracy")
    
    print(f"\n Key Insights:")
    for insight in summary['statistical_insights']:
        print(f"   • {insight}")
    
    return summary

if __name__ == "__main__":
    # Run the test
    summary = test_statistical_engine()
#     summary = {
#     'dataset_info': {...},                    # Basic dataset statistics
#     'best_discriminative_features': [...],    # Top discriminative features
#     'model_performance': {...},               # ML model accuracies
#     'statistical_insights': [...],            # Human-readable insights
#     'analysis_metadata': {...}               # Additional analysis info
# }