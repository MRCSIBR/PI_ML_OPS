import pandas as pd
from fastapi import FastAPI

app = FastAPI()

@app.get("/games")                                              # funciona
async def get_games(developer: str):
    df = pd.read_csv('filter_steam.csv')
    games = df[df['developer'] == developer].to_dict('records')
    return games
    

@app.get("/userdata/{User_id}")
async def userdata(User_id: str):
    user_item_df = pd.read_csv('user_item.csv')
    
    # Check if the User_id exists in the DataFrame
    if User_id in user_item_df['user_id'].values:
        # Filter the DataFrame to get rows matching the User_id and count the items
        user_items_count = user_item_df[user_item_df['user_id'] == User_id]['items_count'].sum()
        return {"User_id": User_id, "items_count": user_items_count}
    else:
        return {"error": f"Usuario '{User_id}' no encontrado"}


@app.get("/countreviews")
async def countreviews(start_date: str, end_date: str):
    # Add your logic here to calculate the desired review count and recommendation percentage
    # Example logic:
    review_count = calculate_review_count(start_date, end_date)
    recommendation_percentage = calculate_recommendation_percentage(start_date, end_date)
    
    return {
        "start_date": start_date,
        "end_date": end_date,
        "review_count": review_count,
        "recommendation_percentage": recommendation_percentage
    }

@app.get("/genre/{genre}")
async def get_genre_ranking(top_n: int = 10):                                   #funciona
    # Extract and preprocess the genre information
    genres_list = filter_steam_df['genres'].str.split(',').explode()
    genres_list = genres_list.str.strip()

    # Count the occurrences of each genre
    genre_counts = genres_list.value_counts()

    # Select the top N genres for the ranking
    top_genres = genre_counts.head(top_n)

    # Create a dictionary to return the ranking
    ranking = {genre: count for genre, count in top_genres.items()}
    return {" rank" :  ranking }
    
    
@app.get("/userforgenre/{genre}")
async def userforgenre(genre: str):
    # Add your logic here to find the top 5 users for the genre
    # Example logic:
    top_users = find_top_users_for_genre(genre)
    
    return top_users

@app.get("/developer/{desarrollador}")
async def developer(desarrollador: str):
    
    # Filter the DataFrame for the specified developer
    developer_df = df[df['developer'] == desarrollador]

    if developer_df.empty:
        return "Developer not found in the dataset"

    # Convert the 'release_date' column to datetime
    developer_df['release_date'] = pd.to_datetime(developer_df['release_date'])

    # Use .loc to set the 'year' column
    developer_df.loc[:, 'year'] = developer_df['release_date'].dt.year

    # Calculate the number of items per year
    items_per_year = developer_df['year'].value_counts().sort_index()

    # Calculate the percentage of free content per year
    total_items_per_year = developer_df.groupby('year')['price'].count()
    free_items_per_year = developer_df[developer_df['price'] == 0].groupby('year')['price'].count()
    percentage_free_per_year = (free_items_per_year / total_items_per_year) * 100

    # Combine the results into a DataFrame
    result_df = pd.DataFrame({'Items Per Year': items_per_year, 'Percentage Free': percentage_free_per_year})

    return result_df
    
@app.get("/sentiment_analysis/")
async def sentiment_analysis(empresa_desarrolladora: str, año_lanzamiento: int):
    # Filter the DataFrame based on developer and release year
    filtered_df = df[(df['developer'] == empresa_desarrolladora) & (df['release_date'].str.extract(r'(\d{4})', expand=False).astype(float) == año_lanzamiento)]
    
    # Check if there are any matching records
    if len(filtered_df) == 0:
        return {"message": "No matching records found."}
    
    # Calculate the sentiment analysis based on your specific criteria (replace this with your own logic)
    # For demonstration purposes, let's assume there's a 'sentiment' column with values 'positive', 'neutral', or 'negative'
    sentiment_counts = filtered_df['sentiment'].value_counts().to_dict()
    
    return sentiment_counts
     
