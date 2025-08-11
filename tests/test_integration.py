"""
Integration tests for complete Phase 1 analysis pipeline.
Tests end-to-end workflows and component interactions.
"""
import pytest
import pandas as pd
from datetime import date
from pathlib import Path
import tempfile
import os

from src.phase1.run import Phase1Analyzer
from src.phase1.outcomes import export_results_to_csv, generate_summary_statistics
from src.common.types import LaunchEvent, AnalysisResults


class TestPhase1AnalyzerIntegration:
    """Test complete Phase 1 analyzer integration."""
    
    def test_load_stock_data_integration(self, temp_data_dir, sample_csv_file):
        """Test stock data loading with real CSV format."""
        analyzer = Phase1Analyzer(data_dir=str(temp_data_dir), results_dir=str(temp_data_dir))
        
        stock_data = analyzer.load_stock_data("test_data.csv", "TEST")
        
        assert len(stock_data.df) > 0
        assert 'Date' in stock_data.df.columns
        assert 'Adj Close' in stock_data.df.columns
        assert 'Volume' in stock_data.df.columns
        assert stock_data.ticker == "TEST"
        assert len(stock_data.data_quality_checks) > 0
    
    def test_analyze_product_launch_integration(self, temp_data_dir, sample_csv_file, mock_env_vars):
        """Test complete product launch analysis."""
        analyzer = Phase1Analyzer(data_dir=str(temp_data_dir), results_dir=str(temp_data_dir))
        
        # Create test event
        test_event = LaunchEvent(
            name="Test Product Integration",
            company="Test Company",
            ticker="TEST", 
            announcement=date(2020, 6, 1),
            release=date(2020, 7, 15),
            category="Technology"
        )
        
        result = analyzer.analyze_product_launch(test_event, "test_data.csv")
        
        assert result is not None
        assert isinstance(result, AnalysisResults)
        assert result.event.name == "Test Product Integration"
        assert result.baseline is not None
        assert result.data_quality_score >= 0.0
        assert result.data_quality_score <= 1.0
        assert result.analysis_timestamp is not None
    
    def test_analyze_product_launch_error_handling(self, temp_data_dir, mock_env_vars):
        """Test error handling in product launch analysis."""
        analyzer = Phase1Analyzer(data_dir=str(temp_data_dir), results_dir=str(temp_data_dir))
        
        # Test with non-existent file
        test_event = LaunchEvent(
            name="Test Product",
            company="Test Company", 
            ticker="TEST",
            announcement=date(2020, 6, 1),
            release=date(2020, 7, 15)
        )
        
        result = analyzer.analyze_product_launch(test_event, "nonexistent.csv", skip_on_error=True)
        
        assert result is None
        assert len(analyzer.errors) > 0
        assert analyzer.errors[0]['product'] == "Test Product"
    
    def test_analyze_product_launch_insufficient_data(self, temp_data_dir, mock_env_vars):
        """Test analysis with insufficient baseline data."""
        # Create minimal dataset
        minimal_data = pd.DataFrame({
            'Date': pd.date_range(start='2020-05-30', end='2020-06-05', freq='D'),
            'Adj Close': [100, 101, 99, 102, 98, 100, 101],
            'Close': [100, 101, 99, 102, 98, 100, 101],
            'High': [102, 103, 101, 104, 100, 102, 103],
            'Low': [98, 99, 97, 100, 96, 98, 99],
            'Open': [99, 100, 100, 101, 99, 99, 100],
            'Volume': [1000000] * 7
        })
        
        # Save minimal CSV
        csv_path = temp_data_dir / "minimal_data.csv"
        with open(csv_path, 'w') as f:
            f.write("Price,Adj Close,Close,High,Low,Open,Volume\n")
            f.write("Ticker,TEST,TEST,TEST,TEST,TEST,TEST\n")
            f.write("Date,,,,,,\n")
        minimal_data.to_csv(csv_path, mode='a', header=False, index=False)
        
        analyzer = Phase1Analyzer(data_dir=str(temp_data_dir), results_dir=str(temp_data_dir))
        
        test_event = LaunchEvent(
            name="Minimal Test",
            company="Test Company",
            ticker="TEST",
            announcement=date(2020, 6, 1),
            release=date(2020, 6, 3)
        )
        
        result = analyzer.analyze_product_launch(test_event, "minimal_data.csv", skip_on_error=True)
        
        # Should either succeed with limited data or fail gracefully
        if result is not None:
            assert result.baseline.trading_days < 60
        else:
            assert len(analyzer.errors) > 0


class TestResultsExportIntegration:
    """Test results export and reporting integration."""
    
    def test_export_results_to_csv_integration(self, temp_data_dir, sample_baseline_metrics, 
                                             sample_signal_metrics, sample_launch_event):
        """Test CSV export with real results data."""
        # Create sample results
        results_list = []
        for i in range(3):
            from src.common.types import StockData
            
            # Create mock stock data for quality checks
            mock_df = pd.DataFrame({
                'Date': pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')[:100],
                'Adj Close': range(100, 200),
                'Volume': [1000000] * 100
            })
            
            mock_stock_data = StockData(
                df=mock_df,
                ticker=f"TEST{i}",
                date_range=(date(2020, 1, 1), date(2020, 12, 31)),
                data_quality_checks={
                    'has_required_columns': True,
                    'no_missing_prices': True, 
                    'no_missing_volumes': True,
                    'no_zero_prices': True,
                    'reasonable_date_gaps': True
                }
            )
            
            from src.phase1.outcomes import compile_analysis_results
            
            result = compile_analysis_results(
                event=LaunchEvent(
                    name=f"Test Product {i}",
                    company="Test Company",
                    ticker=f"TEST{i}",
                    announcement=date(2020, 6, 1),
                    release=date(2020, 7, 15)
                ),
                stock_data=mock_stock_data,
                baseline=sample_baseline_metrics,
                announcement_signal=sample_signal_metrics
            )
            results_list.append(result)
        
        # Export to CSV
        output_path = temp_data_dir / "test_results"
        csv_path, metadata_path = export_results_to_csv(results_list, output_path)
        
        # Verify files were created
        assert csv_path.exists()
        assert metadata_path.exists()
        
        # Verify CSV content
        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert 'product' in df.columns
        assert 'company' in df.columns
        assert 'announcement_5day_return' in df.columns
        
        # Verify metadata content
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert 'analysis_timestamp' in metadata
        assert 'total_products_analyzed' in metadata
        assert metadata['total_products_analyzed'] == 3
    
    def test_generate_summary_statistics_integration(self, sample_baseline_metrics, sample_signal_metrics):
        """Test summary statistics generation with real data."""
        # Create diverse results for statistics
        results_list = []
        
        companies = ["Apple", "Microsoft", "NVIDIA"]
        for i, company in enumerate(companies):
            from src.common.types import StockData
            
            mock_df = pd.DataFrame({
                'Date': pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')[:100],
                'Adj Close': range(100, 200),
                'Volume': [1000000] * 100
            })
            
            mock_stock_data = StockData(
                df=mock_df,
                ticker=f"TEST{i}",
                date_range=(date(2020, 1, 1), date(2020, 12, 31)),
                data_quality_checks={'has_required_columns': True, 'no_missing_prices': True}
            )
            
            # Create varied signal metrics
            signal = sample_signal_metrics
            if i == 1:
                signal.volume_z_score = 3.2  # High significance
            elif i == 2:
                signal.volume_z_score = 1.2  # Low significance
            
            from src.phase1.outcomes import compile_analysis_results
            
            result = compile_analysis_results(
                event=LaunchEvent(
                    name=f"Product {i}",
                    company=company,
                    ticker=f"TEST{i}",
                    announcement=date(2020 + i, 6, 1),
                    release=date(2020 + i, 7, 15)
                ),
                stock_data=mock_stock_data,
                baseline=sample_baseline_metrics,
                announcement_signal=signal
            )
            results_list.append(result)
        
        summary = generate_summary_statistics(results_list)
        
        # Verify summary structure
        assert 'total_products' in summary
        assert 'companies' in summary
        assert 'signal_statistics' in summary
        assert 'data_quality' in summary
        assert 'time_distribution' in summary
        
        # Verify company breakdown
        assert summary['companies']['Apple'] == 1
        assert summary['companies']['Microsoft'] == 1
        assert summary['companies']['NVIDIA'] == 1
        
        # Verify signal statistics
        assert summary['signal_statistics']['announcement_signals']['count'] == 3
        assert 'avg_return' in summary['signal_statistics']['announcement_signals']
        assert 'significant_volume_spikes' in summary['signal_statistics']['announcement_signals']


class TestEndToEndWorkflow:
    """Test complete end-to-end analysis workflow."""
    
    def test_minimal_end_to_end_workflow(self, temp_data_dir, mock_env_vars):
        """Test minimal end-to-end workflow with single product."""
        # Create realistic test data
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        dates = dates[dates.weekday < 5]  # Only weekdays
        
        # Generate realistic price data with some volatility
        np.random.seed(42)
        n_days = len(dates)
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        
        volumes = np.random.lognormal(13, 0.5, n_days).astype(int)  # ~500K average volume
        
        test_data = pd.DataFrame({
            'Date': dates,
            'Adj Close': prices,
            'Close': prices,
            'High': prices * np.random.uniform(1.00, 1.03, n_days),
            'Low': prices * np.random.uniform(0.97, 1.00, n_days),
            'Open': prices * np.random.uniform(0.99, 1.01, n_days),
            'Volume': volumes
        })
        
        # Save test data
        csv_path = temp_data_dir / "end_to_end_test.csv"
        with open(csv_path, 'w') as f:
            f.write("Price,Adj Close,Close,High,Low,Open,Volume\n")
            f.write("Ticker,TEST,TEST,TEST,TEST,TEST,TEST\n")
            f.write("Date,,,,,,\n")
        test_data.to_csv(csv_path, mode='a', header=False, index=False)
        
        # Run analysis
        analyzer = Phase1Analyzer(data_dir=str(temp_data_dir), results_dir=str(temp_data_dir))
        
        test_event = LaunchEvent(
            name="End-to-End Test Product",
            company="Test Company",
            ticker="TEST",
            announcement=date(2020, 6, 1),
            release=date(2020, 7, 15),
            category="Technology"
        )
        
        result = analyzer.analyze_product_launch(test_event, "end_to_end_test.csv")
        
        # Verify successful analysis
        assert result is not None
        assert result.baseline.trading_days > 30  # Should have sufficient baseline
        assert result.announcement_signal is not None
        assert result.release_signal is not None
        
        # Export results
        results_list = [result]
        output_path = temp_data_dir / "end_to_end_results"
        csv_path, metadata_path = export_results_to_csv(results_list, output_path)
        
        assert csv_path.exists()
        assert metadata_path.exists()
        
        # Verify exported data
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.iloc[0]['product'] == "End-to-End Test Product"
        assert pd.notna(df.iloc[0]['baseline_avg_return'])
        assert pd.notna(df.iloc[0]['announcement_5day_return'])
        assert pd.notna(df.iloc[0]['release_5day_return'])