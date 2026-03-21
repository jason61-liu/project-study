package cli

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/shiyiliu/higress-route/internal/validate"
	"github.com/shiyiliu/higress-route/pkg/client"
	"github.com/shiyiliu/higress-route/pkg/route"
)

var (
	deleteName      string
	deleteNamespace string
)

// NewDeleteCmd creates the delete command
func NewDeleteCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "delete",
		Short: "Delete an HTTPRoute",
		Long: `Delete an HTTPRoute resource.

Example:
  hr delete --name my-route --namespace default`,
		RunE: runDelete,
	}

	cmd.Flags().StringVarP(&deleteName, "name", "n", "", "HTTPRoute name (required)")
	cmd.Flags().StringVarP(&deleteNamespace, "namespace", "N", "default", "Kubernetes namespace")

	cmd.MarkFlagRequired("name")

	return cmd
}

func runDelete(cmd *cobra.Command, args []string) error {
	// Validate input
	if err := validate.ValidateName(deleteName); err != nil {
		return err
	}

	if err := validate.ValidateNamespace(deleteNamespace); err != nil {
		return err
	}

	// Create Kubernetes client
	k8sClient, err := client.New(client.Options{Kubeconfig: getKubeconfigPath()})
	if err != nil {
		return fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	// Delete the HTTPRoute
	if err := route.DeleteHTTPRoute(k8sClient, deleteNamespace, deleteName); err != nil {
		return err
	}

	fmt.Printf("HTTPRoute '%s' deleted successfully from namespace '%s'\n", deleteName, deleteNamespace)
	return nil
}
