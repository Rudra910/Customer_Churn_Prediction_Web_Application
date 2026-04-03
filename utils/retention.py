import pandas as pd
import numpy as np
import random as ra
highrisklist = [
    "High Risk: Offer 20% discount coupon + Free delivery on next purchase",
    "High Risk: Provide 30% cashback on next order (limited time)",
    "High Risk: Send win-back email with exclusive deal + countdown timer",
    "High Risk: Offer Buy 1 Get 1 Free on selected products",
    "High Risk: Provide loyalty points boost (2x points on next purchase)",
    "High Risk: Personal call or WhatsApp message with special offer",
    "High Risk: Offer free gift with next purchase",
    "High Risk: Provide subscription discount for 3 months",
    "High Risk: Send 'We miss you' email with personalized discount",
    "High Risk: Give early access to sale + extra 15% off"
]

mediumrisklist = [
    "Medium Risk: Send personalized email with product recommendations + SMS reminder",
    "Medium Risk: Offer small discount (10%) on frequently viewed items",
    "Medium Risk: Send cart reminder with limited-time offer",
    "Medium Risk: Recommend trending products based on browsing history",
    "Medium Risk: Offer loyalty reward points on next purchase",
    "Medium Risk: Send push notification for price drop on wishlist items",
    "Medium Risk: Share customer reviews of previously viewed products",
    "Medium Risk: Offer bundle deals on related products",
    "Medium Risk: Send re-engagement email with new arrivals",
    "Medium Risk: Provide limited-time free delivery offer"
]

lowrisklist = [
    "Low Risk: No action needed - Keep engaging with regular newsletters",
    "Low Risk: Send thank you email with appreciation message",
    "Low Risk: Offer early access to new product launches",
    "Low Risk: Provide exclusive loyalty program benefits",
    "Low Risk: Share referral program with rewards",
    "Low Risk: Send personalized birthday/anniversary offer",
    "Low Risk: Invite to VIP membership or premium plan",
    "Low Risk: Offer sneak peek of upcoming sales",
    "Low Risk: Provide bonus points for social sharing",
    "Low Risk: Engage with interactive content (polls, quizzes)"
]

def get_retention_strategy(probability):
    """Return retention strategy based on churn probability"""
    if probability >= 0.75:
        retention = ra.choices(highrisklist)
        return str(retention[0])
    elif probability >= 0.50:
        retention = ra.choices(mediumrisklist)
        return str(retention[0])
    else:
        retention = ra.choices(lowrisklist)
        return str(retention[0])

def add_retention_strategies(df):
    """Add retention strategy column to dataframe"""
    df['retention_strategy'] = df['churn_probability'].apply(get_retention_strategy)
    return df

def get_churn_status(probability):
    """Convert probability to churn status"""
    return "Yes" if probability >= 0.5 else "No"


def generate_bulk_strategies(probabilities):
    """Generate strategies for multiple probabilities"""
    return [get_retention_strategy(p) for p in probabilities]