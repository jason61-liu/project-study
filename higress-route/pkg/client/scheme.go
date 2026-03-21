package client

import (
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
	gatewayv1beta1 "sigs.k8s.io/gateway-api/apis/v1beta1"
)

// NewScheme creates a new Scheme with all the required types registered
func NewScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()

	// Add the Kubernetes client-go schemes
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))

	// Add Gateway API schemes
	utilruntime.Must(gatewayv1beta1.AddToScheme(scheme))
	utilruntime.Must(gatewayv1.AddToScheme(scheme))

	return scheme
}
