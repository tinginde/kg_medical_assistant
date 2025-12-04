import pandas as pd
import re

class PatientDataLoader:
    """Loads and processes patient data from CSV files."""
    
    def __init__(self, csv_path="mok.csv"):
        self.csv_path = csv_path
        self.df = None
        self.patients = []
    
    def load_csv(self):
        """Load and parse CSV file."""
        self.df = pd.read_csv(self.csv_path)
        return self.df
    
    def parse_weight_change(self, weight_str):
        """Parse weight change string to extract numeric value and category."""
        # Extract numeric value (e.g., "-0.5 kg (Slow)" -> -0.5)
        match = re.search(r'([+-]?\d+\.?\d*)', weight_str)
        if match:
            value = float(match.group(1))
        else:
            value = 0.0
        
        # Extract category
        if 'Slow' in weight_str:
            category = 'Slow'
        elif 'Successful' in weight_str:
            category = 'Successful'
        elif 'Increase' in weight_str:
            category = 'Increase'
        elif 'Moderate' in weight_str:
            category = 'Moderate'
        else:
            category = 'Unknown'
        
        return value, category
    
    def parse_hba1c(self, hba1c_str):
        """Parse HbA1c percentage to float."""
        match = re.search(r'(\d+\.?\d*)', hba1c_str)
        if match:
            return float(match.group(1))
        return 0.0
    
    def parse_numeric_with_commas(self, value):
        """Parse numeric values that may contain commas."""
        if isinstance(value, str):
            value = value.replace(',', '')
        return int(float(value))
    
    def parse_preference(self, pref_str):
        """Parse preference string to extract numeric score."""
        # Extract score from formats like "High (9/10)" or "Low (2/10)"
        match = re.search(r'\((\d+)/10\)', pref_str)
        if match:
            score = int(match.group(1))
        else:
            score = 5  # default
        
        # Extract label
        if 'Very High' in pref_str:
            label = 'Very High'
        elif 'High' in pref_str:
            label = 'High'
        elif 'Medium-High' in pref_str:
            label = 'Medium-High'
        elif 'Medium' in pref_str:
            label = 'Medium'
        elif 'Low' in pref_str:
            label = 'Low'
        else:
            label = 'Unknown'
        
        return score, label
    
    def classify_steps(self, steps):
        """Classify activity level based on daily steps."""
        if steps < 5000:
            return 'Low'
        elif steps < 8000:
            return 'Medium'
        else:
            return 'High'
    
    def classify_caloric_intake(self, calories):
        """Classify caloric intake level."""
        if calories < 1700:
            return 'Low'
        elif calories < 2300:
            return 'Medium'
        else:
            return 'High'
    
    def classify_diet_flexibility(self, score):
        """Classify dietary flexibility preference."""
        if score < 4:
            return 'Low'
        elif score < 8:
            return 'Medium'
        else:
            return 'High'
    
    def process_data(self):
        """Process the loaded CSV data into structured patient records."""
        if self.df is None:
            self.load_csv()
        
        self.patients = []
        
        for _, row in self.df.iterrows():
            # Parse all fields
            weight_value, weight_category = self.parse_weight_change(row['Weight Change (COA)'])
            hba1c = self.parse_hba1c(row['HbA1c (COA)'])
            steps = self.parse_numeric_with_commas(row['Daily Steps (DHT)'])
            calories = self.parse_numeric_with_commas(row['Caloric Intake (DHT)'])
            diet_flex_score, diet_flex_label = self.parse_preference(row['Dietary Flexibility (PPI Preference)'])
            weight_loss_pref_score, weight_loss_pref_label = self.parse_preference(row['Rate of Weight Loss (PPI Preference)'])
            
            patient = {
                'id': row['ID'],
                'weight_change_value': weight_value,
                'weight_change_category': weight_category,
                'hba1c': hba1c,
                'daily_steps': steps,
                'daily_steps_class': self.classify_steps(steps),
                'caloric_intake': calories,
                'caloric_intake_class': self.classify_caloric_intake(calories),
                'diet_flexibility_score': diet_flex_score,
                'diet_flexibility_label': diet_flex_label,
                'diet_flexibility_class': self.classify_diet_flexibility(diet_flex_score),
                'weight_loss_pref_score': weight_loss_pref_score,
                'weight_loss_pref_label': weight_loss_pref_label,
                'scenario': row['Key Scenario (For KG Edges)']
            }
            
            self.patients.append(patient)
        
        return self.patients
    
    def get_patient_by_id(self, patient_id):
        """Get a specific patient by ID."""
        for patient in self.patients:
            if patient['id'] == patient_id:
                return patient
        return None
    
    def get_summary_stats(self):
        """Calculate summary statistics for the dataset."""
        if not self.patients:
            self.process_data()
        
        total = len(self.patients)
        successful = sum(1 for p in self.patients if p['weight_change_category'] == 'Successful')
        slow = sum(1 for p in self.patients if p['weight_change_category'] == 'Slow')
        increase = sum(1 for p in self.patients if p['weight_change_category'] == 'Increase')
        
        avg_weight_change = sum(p['weight_change_value'] for p in self.patients) / total
        avg_steps = sum(p['daily_steps'] for p in self.patients) / total
        avg_calories = sum(p['caloric_intake'] for p in self.patients) / total
        
        return {
            'total_patients': total,
            'successful_count': successful,
            'slow_count': slow,
            'increase_count': increase,
            'success_rate': successful / total * 100,
            'avg_weight_change': avg_weight_change,
            'avg_daily_steps': avg_steps,
            'avg_caloric_intake': avg_calories
        }

if __name__ == "__main__":
    loader = PatientDataLoader("mok.csv")
    patients = loader.process_data()
    
    print(f"Loaded {len(patients)} patients")
    print("\nFirst patient:")
    print(patients[0])
    
    print("\nSummary Statistics:")
    stats = loader.get_summary_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
