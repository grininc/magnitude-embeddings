package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"main/config"
	"math"
	"net/http"
	"sort"
	"time"

	"github.com/weaviate/weaviate-go-client/v4/weaviate"
	"github.com/weaviate/weaviate-go-client/v4/weaviate/fault"
	"github.com/weaviate/weaviate/entities/models"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

const (
	apiKey = config.APIKey
	// mediaLibraryFile = "grin_media_library_content_one.json"
	// mediaLibraryFile = "grin_media_library_content_two.json"
	// mediaLibraryFile = "grin_media_library_content_5000.json"
	// mediaLibraryFile = "grin_media_library_content10000.json"
	// mediaLibraryFile = "grin_media_library_content_500.json"
	// mediaLibraryFile = "grin_media_library_content_all.json"
	//mediaLibraryFile = "mlc_contact_10000.json"
	//mediaLibraryFile = "local1000000.json"
	//mediaLibraryFile = "prod53257.json"
	mediaLibraryFile = "ogall.json"
	endpoint         = "https://api.openai.com/v1/embeddings"
	model            = "text-embedding-ada-002"
	mongoConnection  = "mongodb://root:local@mongodb:27017"
	mongoDatabase    = "admin"
	mongoCollection  = "demo_embeddings"
	// mongoCollection = "embeddings"
	schemaClass  = "ContentCreatorMetaAll"
	createSchema = true
)

type WeaviateClientError struct {
	IsUnexpectedStatusCode bool
	StatusCode             int
	Msg                    string
	DerivedFromError       error
}

func (e *WeaviateClientError) Error() string {
	return e.DerivedFromError.Error()
}

type EmbeddingAPIResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
		Magnitude float64   `json:"magnitude"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

func (r *EmbeddingAPIResponse) SetMagnitude() {
	if len(r.Data) == 0 {
		return
	}
	embedding := r.Data[0].Embedding
	var sum float64
	for _, v := range embedding {
		sum += v * v
	}
	magnitude := math.Sqrt(sum)
	r.Data[0].Magnitude = magnitude
}

type MediaLibraryContentContactMeta struct {
	Id            string  `json:"id"`
	Caption       string  `json:"caption"`
	ContactId     string  `json:"contact_id"`
	NetworkUrl    string  `json:"network_url"`
	Reach         int     `json:"reach"`
	AvgEngagement float64 `json:"avg_engagement"`
}

type MediaLibraryContentContact struct {
	Id         string `json:"id"`
	Caption    string `json:"caption"`
	ContactId  string `json:"contact_id"`
	NetworkUrl string `json:"network_url"`
}

type MediaLibraryContent struct {
	Id       string `json:"id"`
	Caption  string `json:"caption"`
	Hashtags string `json:"hashtags"`
	Mentions string `json:"mentions"`
}

type EmbeddingDocument struct {
	Id        string    `json:"id"`
	Embedding []float64 `json:"embedding"`
	Magnitude float64   `json:"magnitude"`
}

func main() {
	/*
		fmt.Print("Enter your query: ")
		reader := bufio.NewReader(os.Stdin)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		input = strings.TrimSpace(input)

		if input != "init" {
			FindMatches(input)
			return
		}
	*/

	// Read the JSON file into a byte slice
	data, err := ioutil.ReadFile(mediaLibraryFile)
	if err != nil {
		log.Fatalf("Failed to read JSON file: %v", err)
	}

	// Unmarshal the byte slice into a slice of MediaLibraryContent structs
	var content []MediaLibraryContentContactMeta
	err = json.Unmarshal(data, &content)
	if err != nil {
		log.Fatalf("Failed to unmarshal JSON data: %v", err)
	}

	// Compute embeddings for each piece of content
	// var wg sync.WaitGroup
	// var threadLimit = make(chan struct{}, 50) // limit to 100 threads at a time

	cfg := weaviate.Config{
		Host:   "localhost:8080",
		Scheme: "http",
		Headers: map[string]string{
			"X-OpenAI-Api-Key": config.APIKey,
		},
	}

	client, err := weaviate.NewClient(cfg)
	if err != nil {
		panic(err)
	}

	if createSchema {
		classObj := &models.Class{
			Class:      schemaClass,
			Vectorizer: "text2vec-openai", // Or "text2vec-cohere" or "text2vec-huggingface"
		}

		// add the schema
		err = client.Schema().ClassCreator().WithClass(classObj).Do(context.Background())
		if err != nil {
			panic(err)
		}
	}

	objects := make([]*models.Object, len(content))
	for i := range content {
		objects[i] = &models.Object{
			Class: schemaClass,
			Properties: map[string]any{
				"caption":        content[i].Caption,
				"contact_id":     content[i].ContactId,
				"network_url":    content[i].NetworkUrl,
				"media_id":       content[i].Id,
				"reach":          content[i].Reach,
				"avg_engagement": content[i].AvgEngagement,
			},
		}
	}

	// for i := range content {
	// threadLimit <- struct{}{} // add to channel to limit the number of goroutines
	// wg.Add(1)
	// go func(item MediaLibraryContent) {
	// defer wg.Done()
	// defer func() { <-threadLimit }() // remove from channel to allow another goroutine to start
	// batch write items
	totalObjects := len(objects)

	// Define the batch size.
	batchSize := 500

	// Calculate the number of batches.
	numBatches := totalObjects / batchSize
	if totalObjects%batchSize != 0 {
		numBatches++
	}

	// Loop through the objects in batches.
	for i := 0; i < numBatches; i++ {
		// Calculate the start and end indices for the current batch.
		start := i * batchSize
		end := (i + 1) * batchSize
		if end > totalObjects {
			end = totalObjects
		}

		// Extract the objects for the current batch.
		batch := objects[start:end]

		// Use the batch in your Weaviate code.
		res, err := client.Batch().ObjectsBatcher().WithObjects(batch...).Do(context.Background())
		if err != nil {
			fmt.Println()
			var errWeaviate *fault.WeaviateClientError
			errors.As(err, &errWeaviate)
			fmt.Print("Batch error found: ")
			fmt.Println(errWeaviate.DerivedFromError)
		}
		for i := 0; i < len(res); i++ {
			if res[i].Result.Errors != nil {
				fmt.Print("Individual error found: ")
				fmt.Println(res[i].Result.Errors.Error[0].Message)
			}
		}
		fmt.Printf("Batch done: %d\n", (i+1)*batchSize)
		time.Sleep(10000 * time.Millisecond)
	}

	// }(content[i])
	// }
	// wg.Wait()

}

func GetEmbedding(item MediaLibraryContent) EmbeddingAPIResponse {
	client := &http.Client{}
	input := fmt.Sprintf("The caption is as follows: %s. The hashtags are as follows: %s. The mentions are as follows: %s", item.Caption, item.Hashtags, item.Mentions)

	requestData := map[string]interface{}{
		"model": model,
		"input": input,
	}
	requestDataBytes, err := json.Marshal(requestData)
	if err != nil {
		log.Fatal(err)
	}

	req, err := http.NewRequest("POST", endpoint, bytes.NewBuffer(requestDataBytes))
	if err != nil {
		log.Fatal(err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))

	res, err := client.Do(req)
	if err != nil {
		log.Fatal(err)
	}
	defer res.Body.Close()

	var apiResponse EmbeddingAPIResponse
	err = json.NewDecoder(res.Body).Decode(&apiResponse)
	if err != nil {
		log.Fatal(err)
	}

	apiResponse.SetMagnitude()
	return apiResponse
}

func WriteToMongo(item MediaLibraryContent, apiResponse EmbeddingAPIResponse) {
	client, err := mongo.NewClient(options.Client().ApplyURI(mongoConnection))
	if err != nil {
		log.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	err = client.Connect(ctx)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Disconnect(ctx)

	collection := client.Database(mongoDatabase).Collection(mongoCollection)

	embeddingDoc := EmbeddingDocument{
		Id:        item.Id,
		Embedding: apiResponse.Data[0].Embedding,
		Magnitude: apiResponse.Data[0].Magnitude,
	}
	_, err = collection.InsertOne(ctx, embeddingDoc)
	if err != nil {
		log.Fatal(err)
	}
}

func FindMatches(input string) {
	client := &http.Client{}

	requestData := map[string]interface{}{
		"model": model,
		"input": input,
	}
	requestDataBytes, err := json.Marshal(requestData)
	if err != nil {
		log.Fatal(err)
	}

	req, err := http.NewRequest("POST", endpoint, bytes.NewBuffer(requestDataBytes))
	if err != nil {
		log.Fatal(err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))

	res, err := client.Do(req)
	if err != nil {
		log.Fatal(err)
	}
	defer res.Body.Close()

	var apiResponse EmbeddingAPIResponse
	err = json.NewDecoder(res.Body).Decode(&apiResponse)
	if err != nil {
		log.Fatal(err)
	}

	apiResponse.SetMagnitude()
	embeddings, _ := GetAllEmbeddings()
	fmt.Println(len(embeddings))

	var cosineSimilarities []struct {
		ID         string
		Similarity float64
	}

	// Create a channel to synchronize the Goroutines
	ch := make(chan struct{}, 100)

	for _, doc := range embeddings {
		// Add a Goroutine to the channel
		ch <- struct{}{}
		go func(doc EmbeddingDocument) {
			defer func() {
				// Remove the Goroutine from the channel when done
				<-ch
			}()

			// Call cosineSimilarity and append the result to cosineSimilarities
			similarity := cosineSimilarity(doc, apiResponse)
			cosineSimilarities = append(cosineSimilarities, struct {
				ID         string
				Similarity float64
			}{
				ID:         doc.Id,
				Similarity: similarity,
			})
		}(doc)
	}

	// Wait for all Goroutines to finish
	for i := 0; i < cap(ch); i++ {
		ch <- struct{}{}
	}

	// Sort the cosineSimilarities array by magnitude (descending)
	sort.Slice(cosineSimilarities, func(i, j int) bool {
		return cosineSimilarities[i].Similarity > cosineSimilarities[j].Similarity
	})

	// Print the IDs of the top 10 magnitudes
	for i := 0; i < 10 && i < len(cosineSimilarities); i++ {
		if i > 0 {
			fmt.Println(", ")
		}
		fmt.Printf(`'%s'`, cosineSimilarities[i].ID)
	}
	fmt.Println()
}

func GetAllEmbeddings() ([]EmbeddingDocument, error) {
	client, err := mongo.NewClient(options.Client().ApplyURI(mongoConnection))
	if err != nil {
		log.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	err = client.Connect(ctx)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Disconnect(ctx)

	coll := client.Database(mongoDatabase).Collection(mongoCollection)

	// Find all documents in the collection
	cursor, err := coll.Find(context.Background(), bson.D{})
	if err != nil {
		return nil, err
	}

	// Decode each document into an EmbeddingDocument struct and store in an array
	var embeddings []EmbeddingDocument
	for cursor.Next(context.Background()) {
		var embedding EmbeddingDocument
		err := cursor.Decode(&embedding)
		if err != nil {
			return nil, err
		}
		embeddings = append(embeddings, embedding)
	}

	// Check if there were any errors during cursor iteration
	if err := cursor.Err(); err != nil {
		return nil, err
	}

	// Close the cursor and the MongoDB connection
	cursor.Close(context.Background())
	client.Disconnect(context.Background())

	return embeddings, nil
}

func dotProduct(v1, v2 []float64) float64 {
	if len(v1) != len(v2) {
		panic("Vectors must have the same length.")
	}

	result := 0.0
	for i := range v1 {
		result += v1[i] * v2[i]
	}

	return result
}

func cosineSimilarity(doc EmbeddingDocument, input EmbeddingAPIResponse) float64 {
	v1 := doc.Embedding
	v2 := input.Data[0].Embedding
	if len(v1) != len(v2) {
		panic("Vectors must have the same length.")
	}

	dotProd := dotProduct(v1, v2)
	mag1 := doc.Magnitude
	mag2 := input.Data[0].Magnitude

	if mag1 == 0 || mag2 == 0 {
		return 0
	}

	return dotProd / (mag1 * mag2)
}
