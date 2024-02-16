using Plots

# +
using MLJ
using Plots
using StatsBase

struct KNNClassifier
    
    k :: Int
    x_train :: Matrix{Float64}
    y_train :: Vector{Int}

	function KNNClassifier(x, y, k=5)
        
		x_train = copy(x)
        y_train = copy(y)
        
        new(k, x_train, y_train)
    end
    
end


euclidean_distance(x1, x2) = sqrt(sum(x1 .- x2).^2)

function predict_single(model :: KNNClassifier, x)
	# Calculate distances between x and all examples in the training set
	distances = [euclidean_distance(x, p) for p in eachcol(model.x_train)]

	# Get the indices of the k-nearest neighbors
	k_indices = partialsortperm(distances, 1:model.k)

	# Get the labels of the k-nearest neighbors
	k_nearest_labels = [model.y_train[i] for i in k_indices]

	# Return the most common class label among the k-nearest neighbors
	most_common = mode(k_nearest_labels)

	return most_common
    
end

function predict(model::KNNClassifier, x)
	y_pred = [predict_single(model, p) for p in eachcol(x)]
	return y_pred
end


# -


# Generate sample data
n_samples = 4000
n_components = 3
X, y_true = make_blobs(n_samples, 2; centers=n_components, cluster_std=[0.6, 0.6, 0.6])

scatter(X.x1, X.x2, groups = y_true, ms = 1, aspect_ratio=1)

x = stack([X.x1, X.x2], dims = 1)

model = KNNClassifier(x, y_true)

# +
y_pred = predict(model, x)

scatter(X.x1, X.x2, groups=y_pred, aspect_ratio=1, frame=:none)
# -


