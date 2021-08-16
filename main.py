"""
purpose of this project is to try to create a linear regression to
determine/show what features are important to lap times

"""
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from plotly.offline import plot
import plotly.graph_objects as go


def time_string_to_seconds(time_str: str) -> float:
    """ converts a string of time to seconds """
    m, s = time_str.split(":")
    return int(m) * 60 + float(s)


def load_data() -> pd.DataFrame:
    """ load csv of data to train on"""
    return pd.read_csv("data/lapdata_laguna_seca_laurence.csv")


def convert_times(data):
    """ converts string times to seconds """
    columns_to_convert = ["lapTime", "intermediates1", "relativeToStart2"]

    for col in columns_to_convert:
        data[col + 'Seconds'] = [time_string_to_seconds(s) for s in data[col]]

    return data


def extract_per_lap_data(master_data: pd.DataFrame):
    """ extracts data that is the same for all the lap, such as lap time, split times"""
    per_lap_columns = [
        "index", "date", "lapTime", "lapTimeSeconds", "intermediates1", "intermediates1Seconds", "intermediates2"
    ]

    per_lap = master_data[per_lap_columns]

    first_occurrences = []

    for i in per_lap["index"].unique():
        first_occurrences.append(per_lap["index"].eq(i).idxmax())

    return per_lap.loc[first_occurrences]


def remove_outlier_laps(per_lap: pd.DataFrame, percentile: int = 80) -> pd.DataFrame:
    """ removes outlier laps based on lap time """
    cut_off_time = np.percentile(per_lap["lapTimeSeconds"], percentile)
    return per_lap.loc[per_lap['lapTimeSeconds'] <= cut_off_time]


def find_corner_points(point_data, local_points=5, neighbors=2) -> pd.DataFrame:
    """
    analyse the point speeds to determine speed of corner

    local_points is the range around each minimum to choose the minimum
        (assuming value is 5)
        so if points 3 and 5 are minimums we will condense them into one
        but if points 3 and 9 are minimums we decide that they are difference corners
    neighbors are the number of points in each direction to determine noise
    """
    # get minima
    corner_points = list(find_peaks(-1 * point_data["speed"].values, distance=5, prominence=3.5)[0])
    cleaned_corner_points = []
    # remove noise (where all points on left are above and all on right are below, or vise versa)
    for point in corner_points:
        # make sure it's safe to check neighbors
        if point - neighbors < 0 or point + neighbors > len(point_data):
            continue  # if we can't get all the neighbors without over indexing, skip

        point_value = point_data["speed"].values[point]
        left_values = point_data["speed"].values[range(point - neighbors, point)]
        right_values = point_data["speed"].values[range(point + 1, point + neighbors + 1)]
        left_right_values = np.concatenate([left_values, right_values])

        if (
                (max(left_values) > min(right_values) and max(right_values) > min(left_values)) or
                (point_value < min(left_right_values) or point_value > max(left_right_values))
        ):
            cleaned_corner_points.append(point)
    corner_points = cleaned_corner_points

    # combine minimums that are too close to each other
    while True:
        diff = np.diff(corner_points)
        if not any([d < local_points for d in diff]):
            break
        for i, d in enumerate(diff):
            if d < local_points:
                val1 = point_data["speed"].values[corner_points[i]]
                val2 = point_data["speed"].values[corner_points[i+1]]

                if val1 > val2:
                    del corner_points[i]
                else:
                    del corner_points[i+1]
                break

    return point_data.iloc[corner_points]


def label_corners(corner_points: pd.DataFrame):
    """ group the corner points by distance along the course """
    # how many clusters to do?  Gonna go with max corners in any lap
    # get max corners in any lap
    max_number_of_corners = max(corner_points["index"].value_counts())

    # distance from start, time from start, direction
    clustering_columns = [
        'relativeToStart1',
        'relativeToStart2Seconds',
        'direction',
        'speed',
        'coordinate1',
        'coordinate2'
    ]

    kmeans = KMeans(n_clusters=max_number_of_corners).fit(corner_points[clustering_columns])
    corner_points['cornerLabel'] = kmeans.labels_

    corner_identifiers = pd.DataFrame(kmeans.cluster_centers_, columns=clustering_columns)

    # remove corners that have few instances than the 5th percentile
    corner_identifiers['instances'] = corner_points['cornerLabel'].value_counts()
    corner_identifiers = corner_identifiers.loc[
        corner_identifiers['instances'] >= np.percentile(corner_identifiers['instances'], 5)
    ]

    # label the corners by the distance from start, negative one for the zero based index
    corner_identifiers['corner_name'] = corner_identifiers['relativeToStart1'].rank() - 1

    # create map to change corner names from random to order by distance from start
    label_map = {i: int(corner['corner_name']) for i, corner in corner_identifiers.iterrows()}

    number_of_corners = len(label_map)

    corner_points['cornerLabel'] = corner_points["cornerLabel"].map(label_map)
    return corner_points, number_of_corners, corner_identifiers


def create_features(point_data, lap_data):
    """ create single values for each lap from the point data"""

    # extract corners
    corner_points_list = list()
    for lap in lap_data["index"]:
        lap_points = point_data.loc[point_data["index"] == lap]
        corner_points = find_corner_points(lap_points)
        print(f"lap: {lap}, {len(corner_points)} identified corners")
        for i, point in corner_points.iterrows():
            print(f"speed: {point['speed']}, distance: {point['relativeToStart1']}")
        print("\n")
        if len(corner_points) > 7:
            corner_points_list.append(corner_points)

    corner_points = pd.concat(corner_points_list)

    # label the corners (not all laps have the same number of corners
    # label them to allow for consistent referencing
    corner_points, number_of_corners, corner_values = label_corners(corner_points)

    # for each corner get speed
    lap_data.set_index(lap_data['index'], inplace=True)
    for corner in range(0, number_of_corners):
        # assumes corner_points is ordered by index/lap
        lap_n_speeds = corner_points.loc[corner_points['cornerLabel'] == corner, ['index', 'speed']]

        lap_data[f"corner{corner}speed"] = lap_n_speeds.groupby('index')['speed'].apply(lambda x: min(x))

    # TODO: get the straight line speed (later)
    # TODO: pass information back to identify corners
    return lap_data, number_of_corners, corner_values


def remove_laps_with_missing_data(lap_data: pd.DataFrame) -> pd.DataFrame:
    """ removes any laps with missing corner speed data"""
    return lap_data.dropna()


def train_model(laps_data, number_of_corners) -> dict:
    """ trains a linear regression model on the laps data

    returns coefficients for the corners
    """
    corner_numbers = list(range(0, number_of_corners))

    target_col = 'lapTimeSeconds'
    # additional train values could be: ['intermediates1Seconds', 'intermediates2']
    train_cols = [f"corner{i}speed" for i in corner_numbers]

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('regression', LinearRegression())
    ])

    reg = pipe.fit(laps_data[train_cols], laps_data[target_col])

    # coef = pd.DataFrame({c: v for c, v in zip(train_cols, reg.named_steps.regression.coef_)}, index=[0])

    corner_coefs = dict()
    for corner in corner_numbers:
        corner_coefs[corner] = reg.named_steps.regression.coef_[corner]

    return corner_coefs


def add_corner_coefficients_to_corners(corner_data: pd.DataFrame, coef: dict):
    """ add a column to the corner_data df of corner coefficients"""
    # corner_data['coef'] =
    return corner_data


def make_point_elements(point_data: pd.DataFrame):
    """ make point data elements """

    points_scatter = go.Scatter(
        x=point_data.coordinate1 * -1,
        y=point_data.coordinate2,
        marker=dict(size=4, symbol='circle', color='blue'),
        mode='markers',
        name='points',
        hoverinfo='skip'
    )

    return points_scatter


def make_corner_elements(corner_data: pd.DataFrame, corner_coefs: dict):
    """ make corner points """

    corner_elements = list()

    for i, corner in corner_data.iterrows():
        corner_elements.append(
            go.Scatter(
                x=[corner['coordinate1'] * -1],
                y=[corner['coordinate2']],
                marker=dict(size=15, symbol='star', color='orange'),
                mode='markers',
                name=f'corner {int(corner.corner_name)}',
                text=f'corner {int(corner.corner_name)}, coef: {corner_coefs[int(corner.corner_name)]}',
                hoverinfo='text'
            )
        )

    return corner_elements


def plot_corners(point_data: pd.DataFrame, corner_data: pd.DataFrame, corner_coefs: dict):
    """ plot the location of the corners over the point traces so we can see where they actually are"""

    point_elements = make_point_elements(point_data)

    corner_elements = make_corner_elements(corner_data, corner_coefs)

    fig = go.Figure(
        data=[point_elements] + corner_elements,
        layout=go.Layout(
            title='<br>Corner Plots',
            xaxis=dict(
                title='Easting',
                ticklen=5,
                zeroline=False,
                gridwidth=2,
            ),
            yaxis=dict(
                title='Northing',
                ticklen=5,
                gridwidth=2,
                scaleanchor="x",
                scaleratio=1
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=20),
        ))

    fig.update_yaxes(
        scaleanchor='x',
        scaleratio=1
    )

    plot(
        fig,
        filename="Corners Plot.html")


if __name__ == "__main__":

    points = load_data()
    points = convert_times(points)
    laps = extract_per_lap_data(points)
    laps = remove_outlier_laps(laps)

    laps, corner_count, corners = create_features(points, laps)

    laps = remove_laps_with_missing_data(laps)

    corner_coefficients = train_model(laps, corner_count)

    corners = add_corner_coefficients_to_corners(corners, corner_coefficients)

    plot_corners(points, corners, corner_coefficients)

    # plan
    # train multiple modes (there is some randomness in it)
    # in each identify the corners
    # get the average weights for the corners
    # report those and show a plot of the corners

    print("done")
