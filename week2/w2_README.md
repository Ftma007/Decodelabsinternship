# Decodelabs Data Science Internship - Week 2

## Dataset
**E-Commerce Orders Dataset**  
1200 orders with 14 columns including product, payment, referral source, coupon usage, and total price.

## Tasks Completed

### Task 1: Data Collection & Dataset Understanding
- Loaded the Excel dataset using pandas
- Identified all 14 columns and their data types
- Explored unique values in categorical columns (products, payment methods, order statuses)
- Described what the dataset represents

### Task 2: Data Cleaning & Preprocessing
- Found and handled 309 missing values in the CouponCode column
- Confirmed no duplicate rows or duplicate OrderIDs
- Verified data types and fixed the Date column
- Extracted Year, Month, and DayOfWeek features from the Date column
- Detected 8 outliers in TotalPrice using the IQR method

### Task 3: Exploratory Data Analysis (EDA)
- Calculated basic statistics for all numeric columns
- Analyzed revenue by product, payment method, and referral source
- Examined order status distribution
- Compared average order value for coupon vs non-coupon users
- Identified monthly revenue trends

### Task 4: Data Visualization
- Chart 1: Total Revenue by Product (horizontal bar chart)
- Chart 2: Orders by Payment Method (bar chart)
- Chart 3: Orders by Referral Source (bar chart)
- Chart 4: Order Status Distribution (pie chart)
- Chart 5: Monthly Revenue Trend (line chart)
- Chart 6: Distribution of Order Total Price (histogram)
- Chart 7: Average Order Value by Coupon Code (bar chart)

### Task 5: Predictive Model - Linear Regression
- Built a Linear Regression model to predict TotalPrice
- Used features: Quantity, UnitPrice, ItemsInCart, Month, Product, Payment Method, Coupon, Referral Source
- Achieved R-squared score of 0.89 (model explains 89% of price variance)
- Saved actual vs predicted chart for visualization

## Key Findings
- Total revenue across all orders: $1,264,761.96
- Average order value: $1,053.97
- Chair is the top revenue-generating product
- Instagram is the most common referral source
- 74.2% of orders used a coupon code
- Linear Regression model achieved 89% accuracy in predicting order value

## Tools Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
