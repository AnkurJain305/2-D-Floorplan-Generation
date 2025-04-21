from utils import *
from model import *
from upload import *


def get_info():
    """
        Function to return:
            - Boundary: "POLYGON ((105.44810944559121 78 ....
            - front_door: "POLYGON  (105.44810944559121 78 ....
            - room_centroids: [(81, 105), (55, 151), (134, 105)]
            - bathroom_centroids: [(81, 105), (55, 151), (134, 105)]
            - kitchen_centroids: [(81, 105), (55, 151), (134, 105)]
    """
    boundary_wkt = "POLYGON ((25.599999999999994 65.32413793103447, 200.38620689655173 65.32413793103447, 200.38620689655173 75.91724137931033, 230.4 75.91724137931033, 230.4 190.67586206896553, 67.97241379310344 190.67586206896553, 67.97241379310344 176.55172413793102, 25.599999999999994 176.55172413793102, 25.599999999999994 65.32413793103447))"
    
    front_door_wkt = "POLYGON ((38.436315932155225 179.69850789734912, 63.610586007499926 179.69850789734912, 63.610586007499926 176.55172413793102, 38.436315932155225 176.55172413793102, 38.436315932155225 179.69850789734912))"
    
    # Data of the inner rooms or bathrooms
    room_centroids  = [(201, 163), (193, 106)]
    bathroom_centroids = [(91, 91), (52, 95)]
    kitchen_centroids = [(137, 89)]
    
    # boundary_wkt = input("Enter the boundary as str: ")
    # front_door_wkt = input("Enter the front door as str: ")
    # room_centroids = input("Enter the room centroids as list of tuples: ")
    # bathroom_centroids = input("Enter the bathroom centroids as list of tuples: ")
    # kitchen_centroids = input("Enter the kitchen centroids as list of tuples: ")
    
    return boundary_wkt, front_door_wkt, room_centroids, bathroom_centroids, kitchen_centroids


def preProcessing_toGraphs(Boundary, front_door, room_centroids, bathroom_centroids, kitchen_centroids):
    
    Boundary = shapely.wkt.loads(Boundary)
    front_door = shapely.wkt.loads(front_door)
    
    
    # Flip the y axis of all polygons and points
    Boundary = scale(Boundary)
    front_door = scale(front_door)
    room_centroids = [scale(x) for x in room_centroids]
    bathroom_centroids = [scale(x) for x in bathroom_centroids]
    kitchen_centroids = [scale(x) for x in kitchen_centroids]
    
    # Retruning the centroids of the rooms and bathrooms from Point to tuple
    room_centroids = [x.coords[0] for x in room_centroids]
    bathroom_centroids = [x.coords[0] for x in bathroom_centroids]
    kitchen_centroids = [x.coords[0] for x in kitchen_centroids]
            
    living_centroid    = [(Boundary.centroid.x, Boundary.centroid.y)]
    
    user_constraints = {
        'living': living_centroid,
        'room': room_centroids,
        'bathroom': bathroom_centroids,
        'kitchen': kitchen_centroids
    }
    
    # Making networkX graphs [_n for networkX]
    B_n = Handling_dubplicated_nodes(Boundary, front_door)
    G_n = centroids_to_graph(user_constraints, living_to_all=True)
    
    # Converting the networkX graph to pytorch geometric data
    B = from_networkx(B_n, group_node_attrs=['type', 'centroid'], group_edge_attrs=['distance'])
    
    features = ['roomType_embd', 'actualCentroid_x', 'actualCentroid_y']
    G = from_networkx(G_n, group_edge_attrs=['distance'], group_node_attrs=features)
    
    # To use them later for visualization
    
    # Normalization
    G_x_mean = G.x[:, 1].mean().item()
    G_y_mean = G.x[:, 2].mean().item()

    G_x_std = G.x[:, 1].std().item()
    G_y_std = G.x[:, 2].std().item()

    G.x[:, 1:] = (G.x[:, 1:] - torch.tensor([G_x_mean, G_y_mean])) / torch.tensor([G_x_std, G_y_std])
    
    first_column_encodings = F.one_hot(G.x[:, 0].long(), 7)
    G.x = torch.cat([first_column_encodings, G.x[:, 1:]], axis=1)
    
    #### Normalization for the boundary graph
    B_x_mean = B.x[:, 1].mean().item()
    B_y_mean = B.x[:, 2].mean().item()

    B_x_std = B.x[:, 1].std().item()
    B_y_std = B.x[:, 2].std().item()
    
    B.x[:, 1:] = (B.x[:, 1:] - torch.tensor([B_x_mean, B_y_mean])) / torch.tensor([B_x_std, B_y_std])
    
    
    # Befor passing the data to the model
    G.x = G.x.to(torch.float32)
    G.edge_attr = G.edge_attr.to(torch.float32)
    G.edge_index = G.edge_index.to(torch.int64)

    B.x = B.x.to(G.x.dtype)
    B.edge_index = B.edge_index.to(G.edge_index.dtype)
    B.edge_attr = B.edge_attr.to(G.edge_attr.dtype)
    
    
    # Returning back to the original data [not normalized]
    B_not_normalized = B.clone()
    G_not_normalized = G.clone()
    
    G_not_normalized.x[:, -2] = G_not_normalized.x[:, -2] * G_x_std + G_x_mean
    G_not_normalized.x[:, -1] = G_not_normalized.x[:, -1] * G_y_std + G_y_mean

    B_not_normalized.x[:, -2] = B_not_normalized.x[:, -2] * B_x_std + B_x_mean
    B_not_normalized.x[:, -1] = B_not_normalized.x[:, -1] * B_y_std + B_y_mean

    return G, B, G_not_normalized, B_not_normalized, Boundary, front_door, B_n, G_n


    
def Run(Boundary, front_door, room_centroids, bathroom_centroids, kitchen_centroids):
    # Get the data
    # Boundary, front_door, room_centroids, bathroom_centroids, kitchen_centroids = get_info()
    
    # ========================================================================
    # Preprocessing
    G, B, G_not_normalized, B_not_normalized, Boundary_as_polygon, front_door_as_polygon, B_n, G_n = preProcessing_toGraphs(Boundary, front_door, room_centroids, bathroom_centroids, kitchen_centroids)
    the_door = Point(B_not_normalized.x[-1][1:].detach().cpu().numpy()).buffer(3)
    
    # geeing the corresponding graph for the inputs of the user
    
    #=========================================================================
    # Model
    # model_path = r"D:\Grad\Best models\v2\Best_model_V2.pt"
    model_path = r"D:\floor plan\GATNET-model\Checkpoints\best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(model_path, device)
    
    #=========================================================================
    # Inference
    prediction    = model(G.to(device), B.to(device))
    w_predicted   = prediction[0].detach().cpu().numpy()
    h_predicted   = prediction[1].detach().cpu().numpy()
    prediction    = np.concatenate([w_predicted.reshape(-1, 1), h_predicted.reshape(-1, 1)], axis=1)
    
    #=========================================================================
    # Rescaling back to the original values
    # G.x[:, -2] = G.x[:, -2] * G_x_std + G_x_mean
    #=========================================================================
    # Visualization
    output = FloorPlan_multipolygon(G_not_normalized, prediction)
    polygons = output.get_multipoly(Boundary_as_polygon, the_door)
    polygons.plot(cmap='twilight', figsize=(4, 4), alpha=0.8, linewidth=0.8, edgecolor='black');
    
    #=========================================================================
    # Saving the output & Updating to the firebase
    # unique_name = str(uuid.uuid4())
    # if not os.path.exists("./Outputs"):
    #     os.mkdir("Outputs/" + unique_name)
    # plt.savefig("Outputs/" + '/Output.png')
    # image_url = upload_to_firebase(unique_name)
    # print(image_url)
    # print("Done")
    
    
    path = "Outputs/model_output.png"
    plt.savefig(path)
    plt.close()
    return path, B_n, G_n


# if __name__ == '__main__':
#     Run()
# import sys
# import types

# # Workaround for Streamlit + torch watcher error
# class DummyPath:
#     _path = []

# sys.modules['torch.classes'] = types.SimpleNamespace(__path__=DummyPath())
  
# import torch
# from model import load_model
# from utils import parse_prompt, prepare_graph, prepare_dummy_boundary, draw_layout

# def generate_layout_from_prompt(prompt, api_key,checkpoint_path=r"D:\\floor plan\\GATNET-model\\Checkpoints\\best_model.pth"):
#     room_data = parse_prompt(prompt, api_key)
#     if not room_data:
#         print("No room data returned from LLM.")
#         return None

#     graph = prepare_graph(room_data)
#     boundary = prepare_dummy_boundary(graph.num_nodes)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     checkpoint_path=r"D:\floor plan\GATNET-model\Checkpoints\best_model.pth"
#     model = load_model(checkpoint_path, device)

#     graph = graph.to(device)
#     boundary = boundary.to(device)

#     with torch.no_grad():
#         pred_widths, pred_heights = model(graph, boundary)

#     layout_path = r"D:\floor plan\Outputs\generated_layout.png"
#     draw_layout(room_data, pred_widths.cpu().numpy(), pred_heights.cpu().numpy(), save_path=layout_path)
#     return layout_path

# --- Fix for torch.classes crash in Streamlit ---
# ---------- main.py ----------
# import torch
# from model import GATNet, load_model
# from utils import parse_prompt, prepare_graph, prepare_dummy_boundary, draw_layout

# def generate_layout_from_prompt(prompt, api_key, carpet_area=None, checkpoint_path=r"D:\floor plan\GATNET-model\Checkpoints\best_model.pth"):
#     """
#     Generates room layout based on prompt + GATNet model prediction.

#     Args:
#         prompt (str): User description of house
#         api_key (str): OpenAI API key
#         carpet_area (float): Optional total carpet area to guide room sizes
#         checkpoint_path (str): Path to GATNet model weights

#     Returns:
#         layout_path (str): Path to saved floor plan image
#     """
#     if carpet_area:
#         prompt += f"\nThe total carpet area available is {carpet_area} square meters. Keep room sizes within this area."

#     # Step 1: Parse with GPT
#     print("🧠 Parsing prompt with LLM...")
#     room_data = parse_prompt(prompt, api_key, carpet_area=carpet_area)
#     if not room_data:
#         print("❌ No valid room data returned from GPT.")
#         return None
#     print("✅ Parsed Rooms:", room_data)

#     # Step 2: Prepare Graphs
#     graph = prepare_graph(room_data)
#     boundary = prepare_dummy_boundary(graph.num_nodes)

#     # Step 3: Load model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = load_model(device, checkpoint_path)
#     print("✅ GATNet Loaded on:", device)

#     # Step 4: Inference
#     graph = graph.to(device)
#     boundary = boundary.to(device)
#     with torch.no_grad():
#         print("📊 Predicting with GATNet...")
#         pred_widths, pred_heights = model(graph, boundary)
#         print("✅ Prediction Complete")

#     # Step 5: Draw and Save
#     layout_path = draw_layout(room_data, pred_widths.cpu().numpy(), pred_heights.cpu().numpy())
#     print(f"🖼️ Layout saved at: {layout_path}")
#     return layout_path


# # For direct testing
# if __name__ == "__main__":
#     test_prompt = "Design a 2BHK flat with 2 bedrooms, 1 kitchen, 1 hall and 2 bathrooms."
#     your_api_key = ""  # 🔑 Set your OpenAI key here
#     layout = generate_layout_from_prompt(test_prompt, your_api_key, carpet_area=70)
