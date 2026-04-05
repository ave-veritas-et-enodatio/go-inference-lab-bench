package main

import (
	"fmt"
	"log"

	"inference-lab-bench/internal/model"
)

func main() {
	// Test what the manager would do
	dir := "/Users/benn/projects/go-inference-lab-bench/models"
	m, err := model.NewManager(dir)
	if err != nil {
		log.Fatalf("Failed to create manager: %v", err)
	}

	models := m.List()
	fmt.Printf("Found %d models\n", len(models))
	for _, info := range models {
		fmt.Printf("  - %s: arch=%s\n", info.ID, info.Metadata.Architecture)
	}
}
