import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class SyntheticDataGenerator:
    def __init__(self, start_date='2023-01-01', periods=365):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.periods = periods
        
    def generate_base_demand(self):
        """Generate base demand with trend and seasonality"""
        time = np.arange(self.periods)
        
        # Trend
        trend = 1000 + time * 0.5
        
        # Yearly seasonality
        yearly = 200 * np.sin(2 * np.pi * time / 365)
        
        # Weekly seasonality
        weekly = 50 * np.sin(2 * np.pi * time / 7)
        
        return trend + yearly + weekly
    
    def add_promotions(self, base_demand):
        """Add promotion effects"""
        # Generate random promotion dates (about 2 per month)
        promo_dates = np.sort(np.random.choice(
            self.periods, size=24, replace=False
        ))
        
        # Effect lasts for 3-5 days with peak on second day
        promo_effect = np.zeros(self.periods)
        for date in promo_dates:
            duration = np.random.randint(3, 6)
            effect_pattern = np.array([0.5, 1.0, 0.7, 0.3, 0.1])[:duration]
            end_idx = min(date + duration, self.periods)
            promo_effect[date:end_idx] += effect_pattern[:end_idx-date]
        
        return promo_effect * 300, promo_dates
    
    def add_weather_effects(self, base_demand):
        """Add weather effects"""
        # Generate random weather events (extreme temperatures, storms)
        weather_dates = np.sort(np.random.choice(
            self.periods, size=30, replace=False
        ))
        
        weather_effect = np.zeros(self.periods)
        for date in weather_dates:
            # Weather events last 1-3 days
            duration = np.random.randint(1, 4)
            effect = np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.3)
            end_idx = min(date + duration, self.periods)
            weather_effect[date:end_idx] = effect
            
        return base_demand * weather_effect, weather_dates
    
    def add_sports_events(self, base_demand):
        """Add sports event effects"""
        # Generate random sports events (about 2-3 per month)
        sports_dates = np.sort(np.random.choice(
            self.periods, size=30, replace=False
        ))
        
        sports_effect = np.zeros(self.periods)
        for date in sports_dates:
            # Event effect on the day
            sports_effect[date] = np.random.uniform(0.2, 0.4)
            
        return base_demand * sports_effect, sports_dates
    
    def add_school_schedule(self, base_demand):
        """Add school schedule effects"""
        # Define school terms and holidays
        school_effect = np.zeros(self.periods)
        
        # School terms (roughly aligned with typical academic calendar)
        terms = [
            (0, 80),    # Spring term
            (120, 180), # Summer term
            (240, 320)  # Fall term
        ]
        
        for start, end in terms:
            school_effect[start:end] = 0.15
            
        return base_demand * school_effect, [t[0] for t in terms]
    
    def add_holidays(self, base_demand):
        """Add holiday effects"""
        # Major holidays
        holidays = {
            # Approximate days in the year for major holidays
            'new_year': 0,      # Jan 1
            'easter': 90,       # Around Apr 1
            'independence': 185, # July 4
            'thanksgiving': 330, # Late November
            'christmas': 358    # Dec 25
        }
        
        holiday_effect = np.zeros(self.periods)
        holiday_dates = []
        
        for day in holidays.values():
            if day < self.periods:
                # Effect starts a few days before
                start = max(0, day - 3)
                end = min(self.periods, day + 2)
                effect_pattern = np.array([0.2, 0.5, 1.0, 0.5, 0.2])
                holiday_effect[start:end] += effect_pattern[:end-start]
                holiday_dates.append(day)
                
        return base_demand * holiday_effect, sorted(holiday_dates)
    
    def generate_data(self):
        """Generate complete synthetic dataset"""
        # Generate base demand
        base_demand = self.generate_base_demand()
        
        # Add feature effects
        promo_effect, promo_dates = self.add_promotions(base_demand)
        weather_impact, weather_dates = self.add_weather_effects(base_demand)
        sports_impact, sports_dates = self.add_sports_events(base_demand)
        school_impact, school_dates = self.add_school_schedule(base_demand)
        holiday_impact, holiday_dates = self.add_holidays(base_demand)
        
        # Combine all effects
        final_demand = base_demand + promo_effect + weather_impact + \
                      sports_impact + school_impact + holiday_impact
        
        # Add some noise
        noise = np.random.normal(0, base_demand.std() * 0.05, self.periods)
        final_demand += noise
        
        # Create dates
        dates = [self.start_date + timedelta(days=x) for x in range(self.periods)]
        
        # Create DataFrame
        df = pd.DataFrame({
            'ds': dates,
            'y': final_demand,
            'promotions_active': 0,
            'weather_event': 0,
            'sports_event': 0,
            'school_term': 0,
            'holiday': 0
        })
        
        # Mark feature occurrences
        df.loc[promo_dates, 'promotions_active'] = 1
        df.loc[weather_dates, 'weather_event'] = 1
        df.loc[sports_dates, 'sports_event'] = 1
        df.loc[school_dates, 'school_term'] = 1
        df.loc[holiday_dates, 'holiday'] = 1
        
        feature_dates = {
            'promotions': promo_dates.tolist(),
            'weather': weather_dates.tolist(),
            'sports': sports_dates.tolist(),
            'school': school_dates,
            'holidays': holiday_dates
        }
        
        return df, feature_dates

if __name__ == "__main__":
    # Test data generation
    generator = SyntheticDataGenerator()
    df, feature_dates = generator.generate_data()
    print(df.head())
    print("\nFeature dates:", feature_dates)